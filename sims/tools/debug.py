def warn(message):

    # ANSI escape code for red text
    red_code = "\033[91m"

    # ANSI escape code to reset text color
    reset_code = "\033[0m"

    print(f"{red_code}WARNING: {message}{reset_code}")