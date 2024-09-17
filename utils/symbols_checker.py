import re

signs = r'\u0020-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e'
nums=r'\u0030-\u0039'
letters_en=r'\u0041-\u005a\u0061-\u007a'
letters_cyr=r'\u0410-\u044f\u0401\u0451'

emoticons = r'\U0001f600-\U0001f64f' 
dingbats =r'\u2702-\u2714'  # all: r'\u2702-\u27b0' 
transport= r'\U0001f680-\U0001f6a8' 
map_signs=r'\U0001f6a9-\U0001f6c5'
signs_etc=r'\U0001F30D-\U0001F567'

pattern=r'[^'+nums+signs+letters_en+letters_cyr+emoticons+dingbats+transport+map_signs+signs_etc+']'


def match_str(in_str:str):   
    m = re.search(pattern, in_str)
    return m
