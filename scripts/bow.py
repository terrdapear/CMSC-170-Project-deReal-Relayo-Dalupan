VALID_CHAR = set([x for x in 'abcdefghijklmnopqrstuvwxyz1234567890'])

def clean(s: str):
    s_new = "".join([c for c in s.lower() if c in VALID_CHAR])
    return s_new if s_new!='' else None

def clean_update(c, s: str):
    c.update([x for x in map(clean,s.split()) if x!=None])