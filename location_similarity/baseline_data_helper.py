import pandas as pd


def compare_here_poi_in_radius(col_name, data_with_neighbourhood):
    """
    Compare if the poi in radius is consistent with the radius size and locate those streets which are not consisten
    Input:
       radius_suffix: the radius of data we wish to check
       radius_suffix: the index of that radius suffix in the list
    Output:
        street_id of the streets which DOES NOT have consistent number of POIs
    """
    ls_radius_suffix = ['25', '50', '100', '150', '250', '500']
    ls_here_problematic_poi = {}
    for i, j in enumerate(ls_radius_suffix[:-1]):
        ls_here_problematic_poi[f'{col_name}_{ls_radius_suffix[i]}_compare_{ls_radius_suffix[i + 1]}'] = \
            data_with_neighbourhood[
                (data_with_neighbourhood[f'{col_name}_here_{ls_radius_suffix[i]}']
                 > data_with_neighbourhood[f'{col_name}_here_{ls_radius_suffix[i + 1]}'])].street_id.unique()

    return ls_here_problematic_poi


def get_list_street_id(dict_problematic_poi: dict):
    """
    This function take a dictionary of problematic poi and generate a dictionary of lists of streets where the POI is
    problematic
    """
    dict_list_street_ids = {}
    for poi_type, dict_street_ids in dict_problematic_poi.items():
        ls_street_id = []
        for key, street_ids in dict_street_ids.items():
            if not list(street_ids):
                continue
            for street_id in street_ids:
                ls_street_id.append(street_id)
        dict_list_street_ids[poi_type] = ls_street_id

    return dict_list_street_ids


def radius_count_replacement(dict_ls_street_id: dict, data: pd.DataFrame):
    """
    replace the radius column where smaller radius count > than bigger radius count, make sure it is increasing order
    from small radius to bigger radius
    Output:
        the dataframe which has been updated
    """
    for poi_type, ls_street_ids in dict_ls_street_id.items():
        # locate the data
        col_names = [f'{poi_type}_here_25', f'{poi_type}_here_50', f'{poi_type}_here_100', f'{poi_type}_here_150',
                     f'{poi_type}_here_250', f'{poi_type}_here_500']
        for street_id in ls_street_ids:
            located_data = data.loc[data.street_id == street_id, col_names]

            # starting from end of the list, compare column the current index, with the previous column
            for index, col_name in reversed(list(enumerate(col_names))):
                if index == 0:
                    break
                data_min_current = located_data[col_names[index]]
                if (located_data[col_names[index - 1]] > located_data[col_names[index]]).sum() == len(
                        located_data):  # if the current column values are smaller than the previous index's column values
                    located_data[col_names[index - 1]] = located_data[col_names[index]]
                else:
                    data_min_current = located_data[col_names[index - 1]]
            data.loc[data.street_id == street_id, col_names] = located_data

    return data