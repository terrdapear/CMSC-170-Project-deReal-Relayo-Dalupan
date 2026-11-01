
MAGIC_NUM = 3847
with open("repacts_with_summary.txt") as f:
    curr = 0

    i=0
    rawr = []
    for line in f:
        curr+=len(line.split())
        rawr.append(line)
        if curr>MAGIC_NUM:
            with open(f"RA_split/{i:04d}.txt",'w') as g:
                for raaawr in rawr:
                    g.write(raaawr)
                g.close()
            i+=1
            curr=0
            rawr.clear()
    with open(f"RA_split/{i:04d}.txt",'w') as g:
        for raaawr in rawr:
            g.write(raaawr)
        g.close()