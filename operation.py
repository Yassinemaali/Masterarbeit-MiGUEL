import sys
import numpy as np
import datetime as dt
import pandas as pd
from pathlib import Path
# MiGUEL modules
from environment import Environment
from components.pv import PV
from components.windturbine import WindTurbine
from components.storage import Storage
from components.grid import Grid
from components.Electrolyser import Electrolyser
from components.Brennstoffzelle import FuelCell
from components.H2_Storage import H2Storage
import matplotlib.pyplot as plt


class Operator:
    """
    Class to control environment, dispatch dispatch and parameter optimization
    """

    def __init__(self,
                 env: Environment):
        """
        :param env: env.Environment
            system environment
        """
        self.env = env
        self.energy_data = self.env.calc_energy_consumption_parameters()
        self.energy_consumption = self.energy_data[0]
        self.peak_load = self.energy_data[1]
        self.system_covered = None
        self.system = {0: 'Off Grid System', 1: 'Stable Grid connection', 2: 'Unstable Grid connection'}
        self.power_sink = pd.DataFrame(columns=['Time', 'P [W]'])
        self.power_sink = self.power_sink.set_index('Time')
        self.power_sink_max = None
        self.df = self.build_df()
        self.dispatch_finished = False
        self.dispatch()
        self.export_data()

    ''' Basic Functions'''

    def build_df(self):
        """
        Assign columns to pd.DataFrame
        :return: pd.DataFrame
            DataFrame with component columns
        """
        df = pd.DataFrame(columns=['Load [W]', 'P_Res [W]','PV_Production [W]'],
                          index=self.env.time)
        df['Load [W]'] = self.env.df['P_Res [W]'].round(2)
        df['P_Res [W]'] = self.env.df['P_Res [W]'].round(2)
        df['PV_Production'] = self.env.df['PV total power [W]'].round(2)
        #df['P_Res [W]'] = self.env.df['PV total power [W]'].round(2) - self.env.df['P_Res [W]'].round(2)
        #df[['Load [W]', 'P_Res [W]']].to_csv("DF_OPER_1.csv")
        #print("DEBUG: Initialisierung des DataFrame")
        #print("DEBUG: PV-Produktion:", self.env.df['PV total power [W]'].head(5))
        #print("DEBUG: P_Res vor Simulation:", df['P_Res [W]'].head(5).round(2))
        #print("DEBUG: Load:", self.env.df['P_Res [W]'].head(5).round(2))
        # df.to_csv("data frame Operation bevor Batterie und wasserstoff.csv")
        if self.env.grid_connection:
            if self.env.blackout:
                df['Blackout'] = self.env.df['Blackout']
        for pv in self.env.pv:
            pv_col = f'{pv.name} [W]'
            df[pv_col] = 0
            df[f'{pv.name} production [W]'] = pv.df['P [W]']
        for wt in self.env.wind_turbine:
            wt_col = f'{wt.name} [W]'
            df[wt_col] = 0
            df[f'{wt.name} production [W]'] = wt.df['P [W]']
        for es in self.env.storage:
            es_col = f'{es.name} [W]'
            df[es_col] = 0
            df[f'{es.name}_capacity [Wh]'] = np.nan
        for el in self.env.electrolyser:
            el_col = f'{el.name} [W]'
            df[el_col] = 0
            df[f'{el.name} power [W]'] = el.df_electrolyser['P[W]']
        for hstr in self.env.H2Storage:
            hstr_col = f'{hstr.name} [W]'
            df[hstr_col] = 0
            df[f'{hstr.name}: H2 Outflow [kg]'] = hstr.hstorage_df['H2 Outflow [kg]']
            df[f'{hstr.name}: H2 Inflow [kg]'] = hstr.hstorage_df['H2 Inflow [kg]']
            df[f'{hstr.name} _Storage Level [kg]'] = hstr.hstorage_df['Storage Level [kg]']
        #for fc in self.env.fuel_cell:
            #fc_col = f'{fc.name} [W]'
            #df[fc_col] = 0
            #df[f'{fc.name} Power[W]'] = fc.df_fc['Power Output [W]']

        if self.env.grid is not None:  # sicherstellen dass das Grid nur dann in DF AAUFGENOMMEN WIRD;Wenn ein Netz existiert
            grid_col = f'{self.env.grid.name} [W]'
            df[grid_col] = 0

        return df

    ''' Simulation '''

    def dispatch(self):
        """
        dispatch:
        Basic priorities
            1) RE self-consumption
            2) Charge storage from RE
        :return: None
        """
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 10)

        #print(self.df[['Load [W]', 'P_Res [W]', 'PV_Production']].head(100))

        env = self.env
        print(f"DEBUG: Dispatch gestartet")
        processed_times = set()

                # Time step iteration
        for clock in self.df.index:
            for component in env.re_supply:
                # Priority 1: RE self supply

                self.re_self_supply(clock=clock,
                                    component=component)

                # Priority 2: Charge Storage from RE
                for es in env.storage:

                    self.re_charge(clock=clock,
                                   es=es,
                                   component=component)

            # Priority 3: Betrieb Electrolyser
            for el in env.electrolyser:

                self.electrolyser_operate(clock=clock, el=el, component=component)

                h2_produced = el.df_electrolyser.at[clock, 'H2_Production [kg]']


            for h2_storage in env.H2Storage:

                self.H2_charge(clock=clock, hstr=h2_storage, inflow=h2_produced, el=el)

            for fc in env.fuel_cell:
                if self.df.at[clock,'P_Res [W]'] == 0.0:
                   self.df.at[clock, f'{fc.name} [W]'] = 0

            if env.grid_connection is True:
                # system with grid connection
                if env.blackout is False:
                    # stable grid connection
                    self.stable_grid(clock=clock)
                else:
                    # Unstable grid connection
                    self.unstable_grid(clock=clock)
            else:
                # Off grid system
                self.off_grid(clock=clock)

        '''
        


        for el in env.electrolyser:
                el.df_electrolyser.to_csv(f"electrolyser_data_{el.name}.csv", sep=";", decimal=".", index=True)
                print(f"DEBUG: Elektrolyseur-Daten für {el.name} gespeichert in electrolyser_data_{el.name}.csv")
                print(f"DEBUG: die Jährliche Wasserstoff Produktion ist : {el.df_electrolyser['H2_Production [kg]'].sum()} kg")
                #el.plot_electrolyser_power()
        for hstr in env.H2Storage:
            hstr.hstorage_df.to_csv(f"H2 storage dataframe {hstr.name}.csv", sep=";", decimal=".", index=True)
            print(f"DEBUG:H2 Data Frame {hstr.name} gespeichert in h2storage_data_{hstr.name}.csv")
        for es in env.storage:
            es.df.to_csv(f"storage dataframe {es.name}.csv", sep=";", decimal=".", index=True)
            print(f"DEBUG: Storage Data Frame {es.name} wird in storage_data _{es.name}.csv")
        for fc in env.fuel_cell:
            fc.df_fc.to_csv(f" FC DATA {fc.name}.csv",  sep=";", decimal=".", index=True)
        '''


        for pv in self.env.pv:
            col = pv.name + ' [W]'
            self.df[col] = np.where(self.df[col] < 0, 0, self.df[col])

        if self.env.feed_in:
            for component in env.re_supply:
                self.feed_in(component=component)
        power_sink = self.check_dispatch()
        self.power_sink = pd.concat([self.power_sink, power_sink])
        if len(self.power_sink) == 0:
            self.power_sink_max = 0
            self.system_covered = True
        else:
            self.power_sink_max = float(self.power_sink.max().iloc[0])
            self.system_covered = False
        self.dispatch_finished = True
        for fc in env.fuel_cell:
            #self.df= self.df.iloc[0:2000]
            self.df[['Load [W]', 'P_Res [W]', f'{es.name} [W]',f'{es.name} soc', f'{el.name} [W]',f'{el.name} Hydrogen [kg]', f'{h2_storage.name} level [kg]', f'{fc.name} [W]']].to_csv("DF_OPER_2.csv")
            fc.df_fc.to_csv("FC.csv")


        # soc
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df[f'{es.name} soc'], label="State of Charge (SOC)", linewidth=2)
        plt.xlabel("Zeit")
        plt.ylabel("SOC")
        plt.title("SOC-Verlauf über die Zeit")
        plt.legend()
        plt.grid(True)
        plt.show()
        '''
        

        plt.figure(figsize=(12, 6))

        # PV-Produktion
        plt.plot(self.df.index[2000:3000], self.df.iloc[2000:3000]['PV_Production'], label="PV-Produktion [W]", linewidth=2, color= 'blue')

        # Last (Load)
        plt.plot(self.df.index[2000:3000], self.df.iloc[2000:3000]['Load [W]'], label="Last [W]", linewidth=2, color='red')

        # Elektrolyseur-Leistung
        plt.plot(self.df.index[2000:3000], self.df.iloc[2000:3000][f'{el.name} [W]'], label="Elektrolyseur-Leistung [W]", linewidth=2, color='green' )

        # Achsenbeschriftungen und Titel
        plt.xlabel("Zeit")
        plt.ylabel("Leistung [W]")
        plt.title("PV-Produktion, Last und Elektrolyseur-Leistung über die Zeit")

        # Legende und Grid
        plt.legend()
        plt.grid(True)

        # Plot anzeigen
        plt.show()
        '''


    def check_dispatch(self):
        """
        Check if all load is covered with current system components
        :return: None
        """
        power_sink = {}  # speichert die nicht gedeckte Leistung
        for clock in self.df.index:
            if self.df.at[clock, 'P_Res [W]'] > 0:
                power_sink[clock] = self.df.at[clock, 'P_Res [W]']

        power_sink_df = pd.DataFrame(power_sink.items(),
                                     columns=['Time', 'P [W]'])
        print (f"nicht gedeckte Leistung power Sink um {clock} ist {power_sink_df['P [W]']}")

        power_sink_df = power_sink_df.set_index('Time')
        power_sink_df = power_sink_df.round(2)
        print("Nicht gedeckte Leistung", power_sink_df)

        return power_sink_df  # die Werte werden in einer DF gegeben mit nicht gedeckte leistung

    def stable_grid(self,
                    clock: dt.datetime):
        """
        Dispatch strategy from stable grid connection
            Stable grid connection:
                3) Cover residual load from Storage
                4) Cover residual load from Grid
        :param clock: dt.datetime
            time stamp
        :return: None
        """
        env = self.env
        for es in env.storage:
            if self.df.at[clock, 'P_Res [W]'] > 0:
                power = self.df.at[clock, 'P_Res [W]']

                discharge_power = es.discharge(clock=clock,
                                               power=power)
                self.df.at[clock, f'{es.name} [W]'] += float(discharge_power)
                self.df.at[clock, 'P_Res [W]'] += float(discharge_power)
        # Priority 4: Cover load from grid
        if self.env.grid_connection:
            self.grid_profile(clock=clock)

    def unstable_grid(self,
                      clock: dt.datetime):
        """
        Dispatch strategy for unstable grid connection
            No Blackout:
                3) Cover residual load from Grid
            Blackout:
                4.1) Cover load from Storage
                4.2) Cover load from Diesel Generator
        :param clock: dt.datetime
            time stamp
        :return: None
        """
        env = self.env
        if not env.df.at[clock, 'Blackout']:
            self.grid_profile(clock=clock)
        else:
            for es in env.storage:
                if self.df.at[clock, 'P_Res [W]'] > 0:
                    power = self.df.at[clock, 'P_Res [W]']
                    discharge_power = es.discharge(clock=clock,
                                                   power=power)
                    self.df.at[clock, f'{es.name} [W]'] += discharge_power
                    self.df.at[clock, 'P_Res [W]'] += discharge_power
            for fc in env.fuel_cell:
                self.re_fc_operate(clock=clock,
                                fc= fc)

    def off_grid(self,
                 clock: dt.datetime):
        """
        Dispatch strategy for Off-grid systems


        :param clock: dt.datetime
            time stamp
        :return: None
        """
        env = self.env
        p_res = self.df.at[clock, 'P_Res [W]']
        t_step = self.env.i_step

        # Check Energy storage parameters
        storage_power = {}
        storage_capacity = {}

        # discharge storage
        for es in env.storage:
            storage_power[es.name] = es.p_n
            storage_capacity[es.name] = (es.df.at[clock, 'Q [Wh]'] - es.soc_min * es.c) * env.i_step / 60

        power_sum = sum(storage_power.values())
        capacity_sum = sum(storage_capacity.values())

        if p_res == 0:
            return
        #if(p_res < power_sum) and (p_res < capacity_sum):
        if clock != es.df.index[0]:

            if es.df.at[clock - dt.timedelta(minutes=t_step), 'SOC'] > es.soc_min :

                # Discharge storage
                    for es in env.storage:
                        power = self.df.at[clock, 'P_Res [W]']
                        discharge_power = es.discharge(clock=clock, power=power)
                        print (f"Debbug: disccharge Power {discharge_power}")
                        self.df.at[clock, f'{es.name} [W]'] += discharge_power
                        self.df.at[clock, 'P_Res [W]'] += discharge_power

            #for fc in env.fuel_cell:
                #self.df.at[clock, f'{fc.name} [W]'] = 0


            if self.df.at[clock, 'P_Res [W]'] != 0:
                for fc in env.fuel_cell:
                    for hstr in env.H2Storage:
                        fc_power = self.re_fc_operate(clock=clock,
                                                    power=p_res,
                                                    hstr=hstr,
                                                    fc=fc)
            else:
                for fc in env.fuel_cell:
                    fc_power = 0

            self.df.at[clock, f'{fc.name} [W]'] = fc_power


        #for es in env.storage:
            #if self.df.at[clock, 'P_Res [W]'] < 0:
                #power = self.df.at[clock, 'P_Res [W]']
                #discharge_power = es.discharge(clock=clock,
                                                #power=power)
                #self.df.at[clock, f'{es.name} [W]'] += discharge_power
                #self.df.at[clock, 'P_Res [W]'] -= discharge_power
        # for dg in env.diesel_generator:
        #     generator_power = self.dg_profile(clock=clock,
        #                                       dg=dg)
        #     if generator_power > p_res:
        #         power = generator_power - p_res
        #         for es in env.storage:
        #             charge_power = es.charge(clock=clock,
        #                                      power=power)
        #             print(clock, charge_power)
        #             self.df.at[clock, f'{es.name} [W]'] += charge_power
        #             power -= charge_power

    def feed_in(self,
                component: PV or WindTurbine):
        """
        Calculate RE feed-in power and revenues
        :param component: PV/WindTurbine
        :return: None
        """
        if self.env.grid_connection is False:
            pass
        else:
            self.df[f'{component.name} Feed in [W]'] = self.df[f'{component.name} remain [W]']
            if isinstance(component, PV):
                self.df[f'{component.name} Feed in [{self.env.currency}]'] \
                    = self.df[
                          f'{component.name} Feed in [W]'] * self.env.i_step / 60 / 1000 * self.env.pv_feed_in_tariff
            elif isinstance(component, WindTurbine):
                self.df[f'{component.name} Feed in [{self.env.currency}]'] \
                    = self.df[
                          f'{component.name} Feed in [W]'] * self.env.i_step / 60 / 1000 * self.env.wt_feed_in_tariff

    def re_self_supply(self,
                       clock: dt.datetime,
                       component: PV or WindTurbine):
        """
        Calculate re self-consumption
        :param clock: dt.datetime
             time stamp
        :param component: PV/Windturbine
            RE component
        :return: None
        """
        df = self.df

        '''
        # Korrekte Eigenverbrauchsberechnung
        self_consumption = min(df.at[clock, 'Load [W]'], component.df.at[clock, 'P [W]'])
        df.at[clock, f'{component.name} self_supply [W]'] = self_consumption

        # Die Last wird zuerst durch erneuerbare Energie gedeckt
        df.at[clock, 'Load [W]'] -= self_consumption

        # Jetzt berechnen wir den **echten Überschuss**, der zur Batterie oder zum Elektrolyseur gehen kann
        remaining_energy = component.df.at[clock, 'P [W]'] - self_consumption
        df.at[clock, f'{component.name} remain [W]'] = max(0, remaining_energy)
        '''


        df.at[clock, f'{component.name} [W]'] = np.where(
            df.at[clock, 'P_Res [W]'] > component.df.at[clock, 'P [W]'],
            component.df.at[clock, 'P [W]'], df.at[clock, 'P_Res [W]'])
        df.at[clock, f'{component.name} [W]'] = np.where(
            df.at[clock, f'{component.name} [W]'] < 0, 0, df.at[clock, f'{component.name} [W]'])
        df.at[clock, f'{component.name} remain [W]'] = np.where(
           component.df.at[clock, 'P [W]'] - df.at[clock, 'P_Res [W]'] < 0,
            0, component.df.at[clock, 'P [W]'] - df.at[clock, 'P_Res [W]'])
        df.at[clock, 'P_Res [W]'] -= df.at[clock, f'{component.name} [W]']
        if df.at[clock, 'P_Res [W]'] < 0:
            df.at[clock, 'P_Res [W]'] = 0
        #print(
            #f"DEBUG: Nach re_self_supply um {clock} – {component.name} remain: {self.df.at[clock, f'{component.name} remain [W]']}")

        #print("df nach Self_ Supply", self.df.head(100))



    def re_charge(self,
                  clock: dt.datetime,
                  es: Storage,
                  component: PV or WindTurbine):
        """
        Charge energy storage from renewable pv, wind turbine
        :param clock: dt.datetime
            time stamp
        :param es: object
            energy storage
        :param component: object
            re component (pv, wind turbine)
        :return: None
        """
        #print(f"DEBUG: Aufruf von re_charge für {es.name} mit {component.name} um {clock}")
        env = self.env
        index = env.re_supply.index(component)

        if clock == self.df.index[0]:
            if index == 0:
                # Set values for first time step
                es.df.at[clock, 'P [W]'] = 0
                es.df.at[clock, 'SOC'] = es.soc
                es.df.at[clock, 'Q [Wh]'] = es.soc * es.c
        # Charge storage

        charge_power = es.charge(clock=clock,
                                 power=self.df.at[clock, f'{component.name} remain [W]'])
        self.df.at[clock, f'{es.name} [W]'] = charge_power
        self.df.at[clock, f'{component.name}_charge [W]'] = charge_power
        self.df.at[clock, f'{component.name} remain [W]'] -= charge_power
        self.df.at[clock, f'{es.name} soc'] = es.df.at[clock, 'SOC']


        #print(f"charge um {clock} ist {self.df.at[clock, f'{es.name} [W]']}")


        #self.df.at[clock, f'{es.name} SOC'] = es.df.at[clock, 'SOC']

            #print(f"DEBUG: Lade Batterie {es.name} um {clock} mit {charge_power} W und soc {es.df.at[clock, 'SOC']}")

        #else:
            #charge_power = 0

        #if pd.isna(charge_power):
                #print(f"⚠️ WARNUNG: charge_power ist NaN um {clock}!")

            # Falls P_Res danach NaN wird
        #if pd.isna(self.df.at[clock, 'P_Res [W]']):
             #print(f"⚠️ WARNUNG: P_Res ist NaN nach der Batterie-Ladung um {clock}!")
            #print(f"DEBUG: Batterie wird nicht geladen, da P_Res {self.df.at[clock, 'P_Res [W]']} W beträgt")
        #charge_power = es.charge(clock=clock,power=self.df.at[clock, f'{component.name} remain [W]'])
        #print(f"DEBUG: Lade Batterie {es.name} um {clock} mit {charge_power} W")

        #self.df.at[clock, f'{es.name} [W]'] = charge_power
        #self.df.at[clock, f'{component.name}_charge [W]'] = charge_power
        #self.df.at[clock, f'{component.name} remain [W]'] -= charge_power
        #self.df.at[clock, 'P_Res [W]'] -= charge_power
        #print(f"DEBUG: P_Res nach Laden  {clock}: {self.df.at[clock, 'P_Res [W]']} W")





    # for component in env.re_supply:
    # print("DEBUG: Erneuerbare Komponenten in env.re_supply:", [component.name for comp in env.re_supply])

    # power_remain = self.df.at[clock, f'{component.name} remain [W]'].sum()
    # self.df.at[clock,'Power_electrolyser'] = power_remain
    # print(f"DEBUG: {clock} - Verbleibender Überschuss nach Batterie: {power_remain} W")

    def grid_profile(self,
                     clock: dt.datetime):
        """
        Cover load from power grid
        :param clock: dt.datetime
            time stamp
        :return: None
        """
        print ("GRID PROFILE WIRD AUFGERUFEN ")
        df = self.df
        grid = self.env.grid.name
        df.at[clock, f'{grid} [W]'] = self.df.at[clock, 'P_Res [W]']
        df.at[clock, 'P_Res [W]'] = 0

    def export_data(self):
        """
        Export data after simulation
        :return: None
        """
        sep = self.env.csv_sep
        decimal = self.env.csv_decimal
        root = sys.path[1]
        Path(f'{sys.path[1]}/export').mkdir(parents=True, exist_ok=True)
        self.df.to_csv(root + '/export/operator.csv', sep=sep, decimal=decimal)
        self.env.weather_data[0].to_csv(f'{root}/export/weather_data.csv', sep=sep, decimal=decimal)
        self.env.wt_weather_data.to_csv(f'{root}/export/wt_weather_data.csv', sep=sep, decimal=decimal)
        self.env.monthly_weather_data.to_csv(f'{root}/export/monthly_weather_data.csv', sep=sep, decimal=decimal)

    def electrolyser_operate(self, clock: dt.datetime,
                             el: Electrolyser,
                             component: PV or WindTurbine):

        """
        Nutzt überschüssige erneuerbare Energie für den Elektrolyseur.
        :param component:
        :param el:
        :param electrolyser:
        :param clock: dt.datetime
        :param
        :return: None
        """
        #print(f"DEBUG: Starte electrolyser_operate für {el.name} um {clock}")
        env = self.env
        index = env.re_supply.index(component)
        if clock == self.df.index[0]:
            if index == 0:
                el.df_electrolyser.at[clock, 'P[W]'] = 0
                el.df_electrolyser.at[clock, 'P[%]'] = 0
                el.df_electrolyser.at[clock, 'H2_Production [kg]'] = 0

        power = 0

        for component in env.re_supply:
            power += self.df.at[clock, f'{component.name} remain [W]']

        el.run(clock=clock, power=power)

        self.df.at[clock, f'{el.name} [W]'] = el.df_electrolyser.at[clock, 'P[W]']
        self.df.at[clock, f'{el.name} Hydrogen [kg]'] = el.df_electrolyser.at[clock, 'H2_Production [kg]']



        #remain = self.df.at[clock, 'P_Res [W]']
        #print(f"DEBUG: Verfügbare Leistung für Elektrolyseur: {remain} W")
        #power_El = min(remain, el.p_n)
        #print(f"DEBUG: Berechnung von power_El für {clock}: {power_El} W")

        #electrolyser_power = el.run(clock, power=power_El)
        #print(f"DEBUG: Elektrolyseur nimmt {electrolyser_power} W auf")
        #h2_Production = el.calc_H2_production(clock, power=self.power)
        #print(f"DEBUG: um {clock} production Hydrogen ELEKTROLYSEUR OPERATE  {h2_Production} kg")

        #efficiency = el.efficiency_electrolyser(clock, h2_production=h2_Production, power=electrolyser_power)


        #print(f"fDEBUG: production Hydrogen  {h2_Production} kg ")

    def H2_charge(self, clock: dt.datetime, hstr: H2Storage, inflow: float, el= Electrolyser):

        env = self.env

        index = env.H2Storage.index(hstr)
        #self.df.at[clock, f'{hstr.name} level [kg]'] = 0.0

        if clock == self.df.index[0]:

            if index == 0:
                # Set values for first time step
                hstr.hstorage_df.at[clock, 'Storage Level [kg]'] = 0.5 * hstr.capacity
                hstr.hstorage_df.at[clock, 'SOC'] = 0.5    # Der Speicher startet mit der Hälfte der Kapazität
                #hstr.hstorage_df.at[clock, 'Q [Wh]'] =

        hstr.charge(clock=clock, inflow=inflow, el=el)


        self.df.at[clock, f'{hstr.name} [W]'] = hstr.hstorage_df.at[clock, 'Q[Wh]']
        self.df.at[clock, f'{hstr.name} SOC[%]'] = hstr.hstorage_df.at[clock, 'SOC']
        self.df.at[clock, f'{hstr.name} level [kg]'] = hstr.hstorage_df.at[clock,'Storage Level [kg]']


        return

    def re_fc_operate(self,
                      clock: dt.datetime,
                      fc:FuelCell,
                      hstr: H2Storage,
                      power: float):
        print(f"DEBUG um {clock} wird die Methode fc_operate aufgerufen ")
        required_Hydrogen = (power * 0.25) / (33330 * fc.efficiency)  # [kg]  Berechnung der notwendigen Wasserstoffsmenge

        # verfügbare Wasserstoff abrufen
        print(f"DEBUG: Typ von hstr = {type(hstr)}")
        print(f"DEBUG: Attribute von hstr = {dir(hstr)}")


        available_h2 = hstr.hstorage_df.at[clock, 'Storage Level [kg]']

        if pd.isna(available_h2):
            previous_values = hstr.hstorage_df.loc[:clock, 'Storage Level [kg]'].dropna()
            if not previous_values.empty:
                available_h2 = previous_values.iloc[-1]  # Letzten gültigen Wert nehmen
                print(
                    f" WARNUNG {clock}: Kein aktueller Speicherstand gefunden! Nehme vorherigen Wert: {available_h2:.3f} kg")
            else:
                available_h2 = 0  # Falls kein Wert vorhanden ist, setzen wir 0 als Fallback
                print(f" WARNUNG {clock}: Kein vorheriger Wert im Speicher gefunden! Setze 0 kg.")

        #available_h2 = hstr.get_storage_level(hstr.current_level)

        used_Hydrogen = min (required_Hydrogen, available_h2)

        if available_h2 < required_Hydrogen:
            print(
                f" WARNUNG {clock}: Nicht genug H₂ im Speicher! Benötigt: {required_Hydrogen:.3f} kg, Verfügbar: {available_h2:.3f} kg")


        power_generated, hydrogen_consumed = fc.fc_operate(clock= clock, hydrogen_used= used_Hydrogen)

          # Dataframe aktualisieren

        self.df.at[clock, f'{fc.name} [W]'] = power_generated   # Umrechnung in Watt
        self.df.at[clock, 'P_Res [W]'] -= power_generated

         # Aktualisieren des H2-Speichers nach Nutzung

        hstr.discharge(clock=clock, outflow=hydrogen_consumed)

        self.df.at[clock, f'{hstr.name} level [kg]'] = hstr.hstorage_df.at[clock, 'Storage Level [kg]']

        # Debug-Ausgabe
        print(
            f"DEBUG {clock}: {fc.name} sollte {required_Hydrogen:.3f} kg H₂ nutzen, hat {hydrogen_consumed:.3f} kg H₂ verbraucht.")
        print(
            f"DEBUG {clock}: Erzeugte Leistung: {power_generated:.2f} kW, Verfügbarer H₂ nach Entladung: {hstr.hstorage_df.at[clock, 'Storage Level [kg]']:.3f} kg")


        return power_generated








