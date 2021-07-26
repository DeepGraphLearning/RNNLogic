separator = ">" * 30
line = "-" * 30

def time(seconds):
    sec_per_min = 60
    sec_per_hour = 60 * 60
    sec_per_day = 24 * 60 * 60

    if seconds > sec_per_day:
        return "%.2f days" % (seconds / sec_per_day)
    elif seconds > sec_per_hour:
        return "%.2f hours" % (seconds / sec_per_hour)
    elif seconds > sec_per_min:
        return "%.2f mins" % (seconds / sec_per_min)
    else:
        return "%.2f secs" % seconds