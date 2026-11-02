from bisect import insort
from collections import Counter
from scripts.bow import clean_update

class KNN_for_2_colors:
    # A,B are a lists of bag of words
    def __init__(s,A,B,k,p):
        print(f"\nCreating KNN model...")
        s.k = k
        s.p = p

        # a point will be a tuple of frequencies

        s.dictionary = set()
        for bag in A+B:
            s.dictionary|=set(bag.keys())

        s.sorted_dictionary = list(sorted(s.dictionary))

        # (color, coords)
        s.plane = []

        for bag in A:
            s.plane.append( (0, s.pointify(bag)))

        for bag in B:
            s.plane.append((1, s.pointify(bag)))

        print(f"KNN model created")

    #converts a bag to a point
    def pointify(s,bag):
        return tuple([bag.get(key,0) for key in s.sorted_dictionary])

    def d(s,a,b):
        return sum([abs(x1-x2)**s.p for x1,x2 in zip(a,b)])**(1/s.p)

    def nearest_k(s,x):
        # (d, coords)
        nearest = []

        for color, point in s.plane:
            if point==x:
                continue
            insort(nearest,(s.d(x,point),(color,point)))

            if len(nearest)>s.k:
                nearest.pop(-1)
        
        return [i[1] for i in nearest]
    
    def classify(s,x): #x is a bag
        x = s.pointify(x)

        knearest = s.nearest_k(x)
        if 2*sum([a[0] for a in knearest])>s.k:
            return 1
        else:
            return 0

    def classify_paragraph(s,p): #p is the raw text, same as in NB
        bag = Counter()
        clean_update(bag,p)
        return s.classify(bag)