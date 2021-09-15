import holidays
import pandas as pd


def get_holidays_seattle(year):
    seattle_holidays = holidays.CountryHoliday('US', prov=None, state='WA', years=year).get_named('day')
    sundays = allsundays(year)
    holiday_df = pd.DataFrame({'holidays' : seattle_holidays + sundays}).drop_duplicates()
    #we believe that behaviour is different when there are many free days in a row (e.g christmas)
    #so check when there are at least two holiday days consecutivley
    holiday_df['two_consec_free_days'] = (holiday_df.sort_values('holidays')-holiday_df.sort_values('holidays').shift(1))<= pd.Timedelta(days=1)
    return holiday_df.sort_values('holidays')
    
def allsundays(year):
    return pd.date_range(start=str(year), end=str(year+1), 
                         freq='W-SUN').tolist()

def add_extra_time_features(df, year, city='seattle'):
    if city != 'seattle':
        raise NotImplementedError('city has to be seattle')
        
    seattle_holidays = get_holidays_seattle(year)
    data_with_holiday = include_holiday_info(df, seattle_holidays)
    data_with_holiday['month'] = data_with_holiday.observation_interval_start.dt.month
    data_with_holiday['day_of_month'] = data_with_holiday.observation_interval_start.dt.day
    
    return data_with_holiday

    
def include_holiday_info(df, holidays):
    df['holiday'] = df.observation_interval_start.map(lambda x: x.date() in holidays['holidays'].values).astype(int)
    
    time_to_next_df = time_to_next_holiday(df, holidays)
    time_since_last_df = time_since_last_holiday(time_to_next_df, holidays)
    return time_since_last_df

def time_to_next_holiday(df, holidays):
    '''
    Computes the number of days to the next holiday, i.e. special public holidays and sundays 
    '''
    df_sorted = df.sort_values('observation_interval_start')
    temp_df = pd.merge_asof(df_sorted, holidays, left_on = 'observation_interval_start', right_on ='holidays', direction='forward')
    temp_df['time_to_next_holiday'] = temp_df.holidays.dt.date - temp_df.observation_interval_start.dt.date
    temp_df['time_to_next_holiday'] = temp_df['time_to_next_holiday'].map(lambda x: x.total_seconds()/(3600*24))
    df_sorted['time_to_next_holiday'] = temp_df['time_to_next_holiday'].values
    #Compute time to next two day holiday (two consecutivly free days)
    temp_df = pd.merge_asof(df_sorted, holidays[holidays.two_consec_free_days], left_on = 'observation_interval_start', right_on ='holidays', direction='forward')
    temp_df['time_to_next_two_day_holiday'] = temp_df.holidays.dt.date - temp_df.observation_interval_start.dt.date
    temp_df['time_to_next_two_day_holiday'] = temp_df['time_to_next_two_day_holiday'].map(lambda x: x.total_seconds()/(3600*24))
    df_sorted['time_to_next_two_day_holiday'] = temp_df['time_to_next_two_day_holiday'].values
    
    return df_sorted

def time_since_last_holiday(df, holidays):
    '''
    Computes the number of days since the last holiday, i.e. special public holidays and sundays 
    '''
    df_sorted = df.sort_values('observation_interval_start')
    temp_df = pd.merge_asof(df_sorted, holidays, left_on = 'observation_interval_start', right_on ='holidays', direction='backward')
    temp_df['time_since_last_holiday'] =   temp_df.observation_interval_start.dt.date - temp_df.holidays.dt.date
    temp_df['time_since_last_holiday'] = temp_df['time_since_last_holiday'].map(lambda x: x.total_seconds()/(3600*24))
    df_sorted['time_since_last_holiday'] = temp_df['time_since_last_holiday'].values
    #Compute time to since last two day holiday (two consecutivly free days)
    temp_df = pd.merge_asof(df_sorted, holidays[holidays.two_consec_free_days], left_on = 'observation_interval_start', right_on ='holidays', direction='backward')
    temp_df['time_since_last_two_day_holiday'] =   temp_df.observation_interval_start.dt.date - temp_df.holidays.dt.date
    # Compute the days since last holiday minus one since the date in the data is the start of the two day holiday
    temp_df['time_since_last_two_day_holiday'] = temp_df['time_since_last_two_day_holiday'].map(lambda x: x.total_seconds()/(3600*24)-1)
    df_sorted['time_since_last_two_day_holiday'] = temp_df['time_since_last_two_day_holiday'].values

    return df_sorted

