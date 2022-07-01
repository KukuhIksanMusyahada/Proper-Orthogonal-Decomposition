def extractMachAndVF(file: str):
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))