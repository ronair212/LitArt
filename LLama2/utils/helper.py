#
def string_to_bool(input_string):
    normalized_string = input_string.strip().lower()
    
    true_values = ["true", "True", "TRUE" , "T" "yes", "1", "t", "y"]
    
    false_values = ["false", "False" , "FALSE", "F", "0", "f", "n"]
    
    if normalized_string in true_values:
        return True
    elif normalized_string in false_values:
        return False
    else:
        raise ValueError("Input string does not represent a boolean value")
