import sys
import os
import datetime as dt
import pandas as pd
import pvlib
import requests
import urllib
from configparser import ConfigParser
from geopy.geocoders import Nominatim
# MiGUEL Modules
from data.data import DB
from components.pv import PV
from components.windturbine import WindTurbine
from components.dieselgenerator import DieselGenerator
from miguel_test.dieselgenerator import DieselGenerator as test_generator
from components.grid import Grid
from components.storage import Storage
from components.load import Load


class Environment:
    """
    Environment class containing all system components
    Negative power values are power consumption (load, storage)
    Positive power values are power production (PV, DieselGenerator, WindTurbine, Grid, Storage)
    """

    def __init__(self,
                 name: str = None,
                 time: dict = None,
                 economy: dict = None,
                 ecology: dict = None,
                 location: dict = None,
                 grid_connection: bool = None,
                 blackout: bool = False,
                 blackout_data: str = None,
                 feed_in: bool = False,
                 diesel_generator_model: str = 'conventional',
                 weather_data: str = None,
                 csv_sep: str = ',',
                 csv_decimal: str = '.'):
        """
        :param location: dict
            Parameter to create location
            {latitude: float,
             longitude: float,
             altitude: float,
             terrain: str}
        :param time: dict:
            Parameter for time series
            {start: dt.datetime,
             end: dt.datetime,
             step: dt.timedelta,
             timezone: str}
        :param economy: dict
            Parameter for economical calculation
            {d_rate: float,
             lifetime: int,
             electricity_price: float [US$/kWh]
             co2_price: float [US$/t]
             diesel_price: float [US$/l]
             pv_feed_in_tariff: float [US$/kWh]
             wt_feed_in_tariff: float [US$/kWh]
             currency: str}
        :param ecology: dict
            Parameter for ecological calculations
            {co2_diesel: float,
             co2_grid: float}
        :param grid_connection: bool
            System grid connected
        :param feed_in: bool
            feed-in possible
        :param blackout: bool
            Blackout occur
        :param blackout_data: str
            File path blackout data
        :param weather_data: str
            File path weather data
        """
        # Component Container
        self.grid = None
        self.load = None
        self.pv = []
        self.diesel_generator = []
        self.wind_turbine = []
        self.re_supply = []
        self.supply_components = []
        self.storage = []
        # Parameters
        self.name = name
        self.csv_sep = csv_sep
        self.csv_decimal = csv_decimal
        # Time values
        self.t_start = time.get('start')
        self.t_end = time.get('end')
        self.t_step = time.get('step')
        self.timezone = time.get('timezone')
        self.i_step = self.t_step.seconds / 60
        time_parameters = self.create_df()
        self.time_series = time_parameters[0]
        self.time = time_parameters[1]
        self.year = self.t_start.year
        # Location
        self.location = location
        self.longitude = self.location.get('longitude')
        self.latitude = self.location.get('latitude')
        self.altitude = self.get_altitude()
        self.terrain = self.location.get('terrain')
        self.address = self.find_location()
        self.hemisphere = self.address[-1]
        self.seasons = self.find_season()
        # Economy
        if economy is None:
            self.currency = 'US$'
            self.d_rate = 0.03
            self.lifetime = 20  # a
            self.pv_feed_in_tariff = 0.00  # US$/kWh
            self.wt_feed_in_tariff = 0.00  # US$/kWh
            self.electricity_price = 0.0  # US$/kWh
            self.diesel_price = 0  # US$/l
            self.avg_co2_price = 0  # US$//t
        else:
            self.currency = economy.get('currency')
            self.d_rate = economy.get('d_rate')
            self.lifetime = economy.get('lifetime')  # a
            self.electricity_price = economy.get('electricity_price')  # US$//kWh
            self.diesel_price = economy.get('diesel_price')  # US$/l
            self.avg_co2_price = economy.get('co2_price')  # US$/t
            self.pv_feed_in_tariff = economy.get('pv_feed_in_tariff')  # US$//kWh
            self.wt_feed_in_tariff = economy.get('wt_feed_in_tariff')  # US$//kWh
        if ecology is None:
            self.co2_diesel = 0.2665  # kg CO2/kWh
            self.co2_grid = 0  # kg CO2/kWh
        else:
            self.co2_diesel = ecology.get('co2_diesel')
            self.co2_grid = ecology.get('co2_grid')
        # Environment DataFrame
        columns = ['P_Res [W]', 'PV total power [W]', 'WT total power [W]']
        self.df = pd.DataFrame(columns=columns, index=self.time)
        self.df['PV total power [W]'] = 0
        self.df['WT total power [W]'] = 0
        if weather_data is None:
            # Include weather data for remote access
            self.weather_data = self.get_weather_data()
            self.wt_weather_data = self.create_wt_weather_data()
            self.monthly_weather_data = self.create_monthly_weather_data()
        else:
            self.weather_data = pd.read_csv(weather_data)

        # Grid connection
        self.grid_connection = grid_connection
        system = {0: 'Off Grid System', 1: 'On Grid System (stable)', 2: 'On Grid System (unstable)'}
        if self.grid_connection:
            self.add_grid()
            self.blackout = blackout
            if self.blackout:
                self.blackout_data = blackout_data
                blackout_df = pd.read_csv(self.blackout_data, sep=self.csv_sep)
                self.df['Blackout'] = blackout_df['Blackout'].values
                self.system = system[2]
            else:
                self.blackout_data = None
                self.system = system[1]
        else:
            self.system = system[0]
            self.blackout = None
            self.blackout_data = None
        self.feed_in = feed_in
        # Diesel Generator
        self.diesel_generator_model = diesel_generator_model

        # DataBase
        self.database = DB()

        self.supply_data = pd.DataFrame(columns=['Component',
                                                 'Name',
                                                 'Nominal Power [kW]',
                                                 f'Specific investment cost [US$/kW]',
                                                 f'Investment cost [US$]',
                                                 f'Specific operation maintenance cost [US$/kW]',
                                                 f'Operation maintenance cost [US$/a]'])
        self.storage_data = pd.DataFrame(columns=['Component',
                                                  'Name',
                                                  'Nominal Power [kW]',
                                                  'Capacity [kWh]',
                                                  f'Specific investment cost [US$/kWh]',
                                                  f'Investment cost [US$]',
                                                  f'Specific operation maintenance cost [US$/kWh]',
                                                  f'Operation maintenance cost [US$/a]'])

        self.config = ConfigParser()
        self.create_config()

    def find_location(self):
        """
        Find address based on coordinates
        :return: list
        """
        geolocator = Nominatim(user_agent='geoapiExercises')
        location = geolocator.reverse(f'{self.latitude},{self.longitude}')
        if location is None:
            sys.exit('Coordinates not on land.')
        address = location.raw['address']
        city = address.get('city', '')
        if city == '':
            city = None
        state = address.get('state', '')
        country = address.get('country', '')
        code = address.get('country_code')
        zipcode = address.get('postcode')
        if self.latitude < 0:
            hemisphere = 'south'
        else:
            hemisphere = 'north'

        return city, zipcode, state, country, code, hemisphere

    def get_altitude(self):
        """
        Get elevation from coordinates
        :return:
        """
        url = f'https://api.opentopodata.org/v1/aster30m?locations={self.latitude},{self.longitude}'
        result = requests.get(url)

        elevation = result.json()['results'][0]['elevation']

        return elevation

    def find_season(self):
        if self.hemisphere == 'south':
            seasons = {
                'summer': [12, 1, 2],
                'transition': [3, 4, 5, 9, 10, 11],
                'winter': [6, 7, 8],
            }
        else:
            seasons = {
                'winter': [12, 1, 2],
                'transition': [3, 4, 5, 9, 10, 11],
                'summer': [6, 7, 8],
            }
        return seasons

    def create_df(self):
        """
        Create
        :return: list
            time_series,
            df
        """
        time_series = pd.date_range(start=self.t_start,
                                    end=self.t_end,
                                    freq=self.t_step)
        df = pd.Series(time_series)

        return time_series, df

    def get_weather_data(self):
        """
        Retrieve weather data from PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
        :return:
            data: pd.DataFrame
            months_selected: list
            inputs: dict
            metadata: dict
        """
        data, months_selected, inputs, metadata = pvlib.iotools.get_pvgis_tmy(latitude=self.latitude,
                                                                              longitude=self.longitude,
                                                                              startyear=2005,
                                                                              outputformat='json', usehorizon=True,
                                                                              userhorizon=None, map_variables=True,
                                                                              timeout=30,
                                                                              url='https://re.jrc.ec.europa.eu/api/')
        # Set data.index to current year
        current_year = dt.datetime.today().year
        data.index = pd.date_range(start=dt.datetime(
            year=current_year,
            month=1,
            day=1,
            hour=0,
            minute=0),
            end=dt.datetime(
                year=current_year,
                month=12,
                day=31,
                hour=23,
                minute=0),
            freq='1h')

        return data, months_selected, inputs, metadata

    def create_wt_weather_data(self):
        """
        Create weather dataframe
        :return: pd.DataFrame
            wt_data
        """
        # Drop unnecessary columns
        wt_hourly_data = self.weather_data[0].drop(['ghi', 'dni', 'dhi', 'IR(h)'], axis=1)
        # Convert Index
        start_time = dt.datetime(year=self.time_series[0].year,
                                 month=1,
                                 day=1,
                                 hour=0,
                                 minute=0)
        end_time = dt.datetime(year=self.time_series[-1].year,
                               month=12,
                               day=31,
                               hour=23,
                               minute=59)
        wt_hourly_data.index = pd.date_range(start=start_time,
                                             end=end_time,
                                             freq='1h')
        wt_data = wt_hourly_data
        # Interpolate values
        if self.t_step == dt.timedelta(minutes=60):
            pass
        else:
            # Extend index
            wt_data = wt_data.asfreq(self.t_step)
            wt_data.index = wt_data.index.to_pydatetime()
            dates = []
            for i in range(1, int(dt.timedelta(minutes=60) / self.t_step)):
                dates.append(wt_data.index[-1] + pd.Timedelta(i * int(self.t_step / dt.timedelta(minutes=1)), 'min'))
            dates_df = pd.DataFrame(columns=wt_data.columns,
                                    index=dates)
            wt_data = pd.concat([wt_data, dates_df])
            # Interpolate values
            for col in wt_data.columns:
                wt_data[col] = wt_data[col].astype(float)
                wt_data[col] = wt_data[col].interpolate(method='time')

        return wt_data

    def create_monthly_weather_data(self):
        """
        Create monthly weather data
        :return: pd.DataFrame
            monthly weather data
        """
        monthly_weather_data = self.weather_data[0].groupby(lambda x: x.month).mean()

        return monthly_weather_data

    # Add Components to Environment
    def add_grid(self, c_var_n: float = 0):
        """
        Add Grid to environment
        :return: None
        """
        name = 'Grid_1'
        self.grid = Grid(env=self,
                         name=name,
                         c_var_n=c_var_n)
        self.supply_components.append(self.grid)
        self.df[f'{name}: P [W]'] = self.grid.df['P [W]']
        self.df[f'{name}: Blackout'] = self.grid.df['Blackout']
        self.grid_connection = True

    def add_load(self,
                 annual_consumption: float = None,
                 ref_profile: str = None,
                 load_profile: str = None):
        """
        Add Load to environment
        :param: annual_consumption: float
            annual energy demand [kWh]
        :param: load_profile: str
            load profile path
        :return: None
        """
        name = 'Load_1'
        self.load = Load(env=self,
                         name=name,
                         annual_consumption=annual_consumption,
                         ref_profile=ref_profile,
                         load_profile=load_profile)
        self.df['P_Res [W]'] = self.load.df['P [W]']

    def add_pv(self,
               p_n: float = None,
               pv_data: dict = None,
               pv_profile: pd.Series = None,
               c_invest: float = None,
               c_op_main: float = None,
               c_var_n: float = 0):
        """
        Add PV system to environment
        :return: None
        """
        name = f'PV_{len(self.pv) + 1}'
        if pv_profile is not None:
            self.pv.append(PV(env=self,
                              name=name,
                              pv_profile=pv_profile,
                              c_invest=c_invest,
                              c_op_main=c_op_main,
                              c_var_n=c_var_n))
        elif p_n is not None:
            self.pv.append(PV(env=self,
                              name=name,
                              p_n=p_n,
                              pv_data=pv_data,
                              c_invest=c_invest,
                              c_op_main=c_op_main,
                              c_var_n=c_var_n))
        elif pv_data is not None:
            self.pv.append(PV(env=self,
                              name=name,
                              pv_data=pv_data,
                              c_invest=c_invest,
                              c_op_main=c_op_main,
                              c_var_n=c_var_n))
        else:
            pass
        self.re_supply.append(self.pv[-1])
        self.supply_components.append(self.pv[-1])
        self.df[f'{name}: P [W]'] = self.pv[-1].df['P [W]']
        self.df['PV total power [W]'] += self.df[f'{name}: P [W]']
        self.add_component_data(component=self.pv[-1],
                                supply=True)

    def add_wind_turbine(self,
                         p_n: float = None,
                         turbine_data: dict = None,
                         wt_profile: pd.Series = None,
                         selection_parameters: list = None,
                         c_invest: float = None,
                         c_op_main: float = None,
                         c_var_n: float = 0):
        """
        Add Wind Turbine to environment
        :return: None
        """
        name = f'WT_{len(self.wind_turbine) + 1}'
        self.wind_turbine.append(WindTurbine(env=self,
                                             name=name,
                                             p_n=p_n,
                                             turbine_data=turbine_data,
                                             wt_profile=wt_profile,
                                             selection_parameters=selection_parameters,
                                             c_invest=c_invest,
                                             c_op_main=c_op_main,
                                             c_var_n=c_var_n))
        self.re_supply.append(self.wind_turbine[-1])
        self.supply_components.append(self.wind_turbine[-1])
        self.df[f'{name}: P [W]'] = self.wind_turbine[-1].df['P [W]']
        self.df['WT total power [W]'] += self.df[f'{name}: P [W]']
        # self.add_component_data(component=self.wind_turbine[-1], supply=True)

    def add_diesel_generator(self,
                             p_n: float = None,
                             model: bool = False,
                             c_invest: float = None,
                             c_op_main: float = None,
                             c_var_n: float = 0):
        """
        Add Diesel Generator to environment
        :return: None
        """
        name = f'DG_{len(self.diesel_generator) + 1}'
        self.diesel_generator.append(DieselGenerator(env=self,
                                                     name=name,
                                                     p_n=p_n,
                                                     model=model,
                                                     c_invest=c_invest,
                                                     c_op_main=c_op_main,
                                                     c_var_n=c_var_n))
        self.df[f'{name}: P [W]'] = self.diesel_generator[-1].df['P [W]']
        self.supply_components.append(self.diesel_generator[-1])
        self.add_component_data(component=self.diesel_generator[-1],
                                supply=True)

    def add_storage(self,
                    p_n: float = None,
                    c: float = None,
                    soc: float = 0.25,
                    soc_max: float = 0.95,
                    soc_min: float = 0.05,
                    lifetime: int = 10,
                    c_invest: float = None,
                    c_op_main: float = None,
                    c_var_n: float = 0.021):
        """
        Add Energy Storage to environment
        :return: None
        """
        name = f'ES_{len(self.storage) + 1}'
        self.storage.append(Storage(env=self,
                                    name=name,
                                    p_n=p_n,
                                    c=c,
                                    soc=soc,
                                    soc_min=soc_min,
                                    soc_max=soc_max,
                                    lifetime=lifetime,
                                    c_invest=c_invest,
                                    c_op_main=c_op_main,
                                    c_var_n=c_var_n))
        self.df[f'{name}: P [W]'] = self.storage[-1].df['P [W]']
        self.add_component_data(component=self.storage[-1],
                                supply=False)

    def add_component_data(self,
                           component,
                           supply: bool):
        """
        Add technical data of component to component df
        :param supply: bool
        :param component: object
            Object of create Component
        :return: None
        """
        if supply is True:
            self.supply_data = self.supply_data._append(component.technical_data,
                                                        ignore_index=True)
        else:
            self.storage_data = self.storage_data._append(component.technical_data,
                                                          ignore_index=True)

    def calc_energy_consumption_parameters(self):
        """
        Calculate total energy consumption and peak load
        :return: list
            energy_consumption [kWh], peak_load [W]
        """
        energy_consumption = self.df['P_Res [W]'].sum() * self.i_step / 60 / 1000
        peak_load = self.df['P_Res [W]'].max()

        return energy_consumption, peak_load

    def create_config(self):
        """
         Create and write config file for system configuration
         :return: None
         """
        self.config[self.name] = {'latitude': str(self.latitude),
                                  'longitude': str(self.longitude),
                                  'altitude': str(self.altitude),
                                  'terrain': str(self.terrain),
                                  't_start': str(self.t_start),
                                  't_end': str(self.t_end),
                                  't_step': str(self.t_step),
                                  'tz': str(self.timezone),
                                  'grid_connection': str(self.grid_connection),
                                  'blackout': str(self.blackout),
                                  'blackout_data': str(self.blackout_data),
                                  'feed_in': str(self.feed_in),
                                  'currency': str(self.currency),
                                  'lifetime': str(self.lifetime),
                                  'd_rate': str(self.d_rate),
                                  'electricity_price': str(self.electricity_price),
                                  'diesel_price': str(self.diesel_price),
                                  'wt_feed_in_tariff': str(self.wt_feed_in_tariff),
                                  'pv_feed_in_tariff': str(self.pv_feed_in_tariff),
                                  'co2_grid': str(self.co2_grid),
                                  'co2_diesel': str(self.co2_diesel)}

        path = f'{sys.path[1]}/export/config/'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}system_config.ini', 'w') as file:
            self.config.write(file)
