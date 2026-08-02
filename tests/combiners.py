from jerrythomas.domain.record import TemporalRecord


def combine_valuation_inputs(price, earnings) -> TemporalRecord:
    record = TemporalRecord(time=price.time)
    record.ticker = price.ticker
    record.price = price.value
    record.earnings = earnings.value
    return record


def combine_humidity_with_baseline(humidity, baseline) -> TemporalRecord:
    record = TemporalRecord(time=humidity.time)
    record.location = humidity.location
    record.humidity = humidity.value
    record.baseline = baseline.value
    record.value = humidity.value + baseline.value
    return record
