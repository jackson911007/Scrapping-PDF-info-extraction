# -*- coding: utf-8 -*-
"""
Function to transform the records extracted via te_parse into a database.
"""
import logging
import pandas as pd  # <-- Make sure pandas is imported

def add_record_to_database(record, df, filename, title, parser, excerpt='-'):
    """Adds record to the DataFrame database, according to expected structure."""
    
    # set everything clean
    compound, editing, temperature, compound_name, label, confidence, synthesis, editing, key, specifier, value, units, \
    temp_value, temp_units, room_temperature, pressure, pressure_value, pressure_units, process, direction_of_measurement, raw_units, raw_value, \
    error, contextual_label = ['-'] * 24

    for model_name in record:
        model = record[model_name]

        compound = model.get('compound', compound)
        compound_name = compound.get('Compound', {}).get('names', compound_name)
        label = compound.get('Compound', {}).get('labels', label)
        synthesis = model.get('synthesis', synthesis)
        editing = model.get('editing', editing)
        process = model.get('process', process)
        direction_of_measurement = model.get('direction_of_measurement', direction_of_measurement)
        confidence = model.get('confidence', confidence)
        specifier = model.get('specifier', specifier)
        value = model.get('value', value)
        units = model.get('units', units)
        raw_value = model.get('raw_value', raw_value)
        raw_units = model.get('raw_units', raw_units)
        error = model.get('error', error)
        contextual_label = model.get('contextual_label', contextual_label)
        room_temperature = model.get('room_temperature', room_temperature)

        # temperature
        temperature_model = model.get('temperature', {})
        temperature = temperature_model.get('Temperature', temperature_model.get('Temperature_for_table_captions', temperature))

        if temperature:
            temp_value = temperature.get('value', temp_value)
            temp_units = temperature.get('units', temp_units)

        # pressure
        pressure_model = model.get('pressure', {}).get('Pressure', {})
        pressure_value = pressure_model.get('value', pressure_value)
        pressure_units = pressure_model.get('units', pressure_units)

    dictionary = {
        'compound_name': compound_name, 'labels': label, 'editing': editing, 'model': model_name,
        'specifier': specifier, 'raw_units': raw_units, 'raw_value': raw_value, 'error': error,
        'value': value, 'units': units, 'temp_value': temp_value, 'temp_units': temp_units,
        'room_temperature': room_temperature, 'process': process, 'direction_of_measurement': direction_of_measurement,
        'excerpt': excerpt, 'filename': filename, 'title': title, 'parser': parser,
        'pressure_value': pressure_value, 'pressure_units': pressure_units
    }

    # ✅ FIXED append to be compatible with pandas >= 2.0
    df = pd.concat([df, pd.DataFrame([dictionary])], ignore_index=True)
    return df
