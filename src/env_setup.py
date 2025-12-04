import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class RealEstatePortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, property_data_path, market_data_path, initial_cash=100_000_000, num_properties=5):
        super().__init__()
        # Load property-level data (tab-separated)
        self.property_df = pd.read_csv(property_data_path, sep='\t', parse_dates=['date'])
        # Convert to monthly period for convenience
        self.property_df['year_month'] = self.property_df['date'].dt.to_period('M')
        
        # Load market-level data (try tab-separated first, then comma)
        try:
            self.market_df = pd.read_csv(market_data_path, sep='\t')
            if self.market_df.empty:
                raise ValueError("Market data file is empty")
        except (pd.errors.EmptyDataError, ValueError, Exception):
            try:
                self.market_df = pd.read_csv(market_data_path)
                if self.market_df.empty:
                    raise ValueError("Market data file is empty")
            except:
                # Create empty market data structure if file doesn't exist or is empty
                print(f"Warning: Could not load market data from {market_data_path}. Creating empty structure.")
                self.market_df = pd.DataFrame(columns=['date', 'unemployment_rate', 'mortgage_rate', 'zillow_index', 'case_shiller_index'])
        # Convert market_df 'date' (YYYY-MM) to a Period
        self.market_df['year_month'] = self.market_df['date'].apply(lambda x: pd.Period(x, freq='M'))

        # Extract all unique year_months from property data
        self.dates = self.property_df['year_month'].unique()
        self.dates = pd.Series(self.dates).sort_values()
        self.current_idx = 0

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.num_properties = num_properties
        self.owned_properties = {}
        self.last_known_prices = {}  # Dictionary to store last known price for each zpid

        # Action space: For each property: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.MultiDiscrete([3]*self.num_properties)

        self.prev_portfolio_value = initial_cash

        # Define observation space dimensions:
        # Example: obs = [cash, unemployment_rate, mortgage_rate, zillow_index, case_shiller_index] + property features
        market_feature_count = 4  # unemployment, mortgage, zillow, case_shiller
        prop_feature_count = 5    # e.g. log_price, bedrooms, bathrooms, livingArea, propertyType encoded
        total_dim = 1 + market_feature_count + self.num_properties*prop_feature_count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.mean_price = self.property_df['price'].mean()
        self.std_price = self.property_df['price'].std()
        self.unemployment_mean = self.market_df['unemployment_rate'].mean()
        self.unemployment_std = self.market_df['unemployment_rate'].std()

        # Initialize variables for tracking returns
        self.returns = []  # List to store incremental profits
        self.epsilon = 1e-8  # Small value to prevent division by zero

    def _encode_property_type(self, ptype):
        mapping = {'SINGLE_FAMILY': 1.0, 'APARTMENT': 2.0, 'MULTI_FAMILY': 3.0}
        return mapping.get(ptype, 0.0)

    def _get_current_data(self):
        current_period = self.dates[self.current_idx]
        subset = self.property_df[self.property_df['year_month'] == current_period]

        if subset.empty:
            print("No properties available this month!")

        top_props = subset.nlargest(self.num_properties, 'price')
        return top_props, current_period

    def _get_observation(self):
        top_props, current_period = self._get_current_data()
        # Get market data for current_period
        market_row = self.market_df[self.market_df['year_month'] == current_period]
        if len(market_row) == 0:
            # If no market data for this period, fill defaults or handle gracefully
            unemployment = 0.0
            mortgage = 0.0
            zillow = 1.0
            cs = 1.0
        else:
            market_row = market_row.iloc[0]
            unemployment = market_row['unemployment_rate']
            mortgage = market_row['mortgage_rate']
            zillow = market_row['zillow_index']
            cs = market_row['case_shiller_index']

        obs = [self.cash, unemployment, mortgage, zillow, cs]

        for _, p in top_props.iterrows():
            log_price = (np.log1p(p['price']) - np.log1p(self.mean_price)) / (np.log1p(self.std_price) if self.std_price != 0 else 1)
            bdr = p['bedrooms'] if not np.isnan(p['bedrooms']) else 0.0
            bth = p['bathrooms'] if not np.isnan(p['bathrooms']) else 0.0
            lv = p['livingArea'] if not np.isnan(p['livingArea']) else 0.0
            pt = self._encode_property_type(p['propertyType'])
            obs.extend([log_price, bdr, bth, lv, pt])

        # If fewer than num_properties, pad
        needed = self.observation_space.shape[0] - len(obs)
        obs.extend([0.0]*needed)
        obs = np.array(obs, dtype=np.float32)
        
        # Debugging: Print the observation to identify NaN values
        #print("Observation before NaN check:", obs)

        # Check for NaN values and handle them
        if np.isnan(obs).any():
            #print("NaN detected in observation:", obs)
            # Handle NaN values, e.g., replace with zero or another default value
            obs = np.nan_to_num(obs, nan=0.0)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.cash = self.initial_cash
        self.owned_properties = {}
        self.last_known_prices = {}
        self.prev_portfolio_value = self.initial_cash
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        top_props, current_period = self._get_current_data()
        #print(f"Current period: {current_period}, Selected Properties: {len(top_props)}")
        #print("Action taken:", action)

        # Execute actions
        for i, act in enumerate(action):
            if i >= len(top_props):
                break
            p = top_props.iloc[i]
            zpid = p['zpid']
            price = p['price']
            #print(f"Property {p['zpid']} priced at {p['price']}. Action: {act}")

            # Store the last known price
            self.last_known_prices[zpid] = price

            if act == 1: # buy
                if self.cash >= price:
                    self.cash -= price
                    self.owned_properties[zpid] = self.owned_properties.get(zpid, 0) + 1
            elif act == 2: # sell
                if zpid in self.owned_properties and self.owned_properties[zpid] > 0:
                    self.owned_properties[zpid] -= 1
                    self.cash += price
                    if self.owned_properties[zpid] == 0:
                        del self.owned_properties[zpid]

        self.current_idx += 1
        done = (self.current_idx >= len(self.dates)-1)

        # Calculate portfolio value
        portfolio_value = self.cash
        if self.owned_properties:
            next_period = self.dates[self.current_idx] if not done else current_period
            subset = self.property_df[self.property_df['year_month'] == next_period]

            for zpid, qty in self.owned_properties.items():
                # If property not found this month, use last known price
                val = subset.loc[subset['zpid'] == zpid, 'price']
                if len(val) > 0:
                    prop_price = val.iloc[0]
                    self.last_known_prices[zpid] = prop_price  # update last known
                else:
                    # Use last known price if we have it, else assume price unchanged
                    prop_price = self.last_known_prices.get(zpid, 0)

                portfolio_value += prop_price * qty
                
        # simple incremental reward 
        reward = portfolio_value - self.prev_portfolio_value # incremental profit/loss
        #---#
        # proportional percentage reward
        # reward = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        #---#
        # sharpe reward: track a running variance or standard deviation of returns
        # Calculate incremental profit/loss
        # incremental_profit = portfolio_value - self.prev_portfolio_value
        # self.returns.append(incremental_profit)
        # std_of_returns = np.std(self.returns)
        # reward = incremental_profit / (std_of_returns + self.epsilon)
        # self.prev_portfolio_value = portfolio_value
        #---#
        # sparsity penalty (transaction costs)
        # cost_per_trade = 100000  # Define a cost per trade
        # buy_sell_actions = np.array(action)  # Convert action to a numpy array
        # trade_cost = np.sum(buy_sell_actions) * cost_per_trade  # Calculate trade cost based on actions
        # reward = (portfolio_value - self.prev_portfolio_value) - trade_cost

        obs = None if done else self._get_observation()
        info = {"portfolio_value": portfolio_value}
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass