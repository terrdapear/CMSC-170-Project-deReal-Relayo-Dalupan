from heapq import nsmallest
from collections import Counter
from scripts.bow import clean_update

class KNN_for_2_colors:
    # A,B are a lists of bag of words
    def __init__(s,A,B,k,p):
        print(f"\nCreating KNN model...")
        s.k = k
        s.p = p

        # a point is a bag, words not in the bag count as 0
        # (color, bag)
        s.plane = []

        for bag in A:
            s.plane.append((0, bag))

        for bag in B:
            s.plane.append((1, bag))

        # sum of |count|^p of each point, its distance^p from an empty bag
        s.sizes = [s.size(bag) for color, bag in s.plane]

        # word -> list of (point index, count), to find points that share a word
        s.index = {}
        for i in range(len(s.plane)):
            for word, count in s.plane[i][1].items():
                s.index.setdefault(word, []).append((i, count))

        # points sorted by size, for points that share no words with x
        s.by_size = sorted(range(len(s.plane)), key=lambda i: (s.sizes[i], s.plane[i][0]))

        print(f"KNN model created")

    def size(s,bag):
        return sum([abs(count)**s.p for count in bag.values()])

    # distance^p from x to every point that shares a word with x
    # same as summing |x1-x2|^p over all words, since the other words add 0
    def d_shared(s,x):
        x_size = s.size(x)
        dists = {}
        for word, x1 in x.items():
            for i, x2 in s.index.get(word, []):
                if i not in dists:
                    dists[i] = x_size + s.sizes[i]
                # swap |x1|^p + |x2|^p for |x1-x2|^p
                dists[i] += abs(x1-x2)**s.p - abs(x1)**s.p - abs(x2)**s.p
        return x_size, dists

    def nearest_k(s,x):
        x_size, dists = s.d_shared(x)

        # (d^p, color, index), no need for the root since it keeps the order
        nearest = []
        for i, d in dists.items():
            # d is 0 only if the point is the same bag as x
            if d==0:
                continue
            nearest.append((d, s.plane[i][0], i))

        # the other points share no words, so the smallest ones are the nearest
        added = 0
        for i in s.by_size:
            if added==s.k:
                break
            if i in dists:
                continue
            d = x_size + s.sizes[i]
            if d==0:
                continue
            nearest.append((d, s.plane[i][0], i))
            added += 1

        # ties go to the lower color, same as before
        return [(color, i) for d, color, i in nsmallest(s.k, nearest)]

    def classify(s,x): #x is a bag
        knearest = s.nearest_k(x)
        if 2*sum([a[0] for a in knearest])>s.k:
            return 1
        else:
            return 0

    def classify_paragraph(s,p): #p is the raw text, same as in NB
        bag = Counter()
        clean_update(bag,p)
        return s.classify(bag)