
def jaccard_index(s1,s2):
  A=set(s1.split())
  B=set(s2.split())
  try:
    return len(A.intersection(B))/len(A.union(B))
  except ZeroDivisionError:
    return 0
    
def cwc(s1,s2,func):
  A=set(s1.split())
  B=set(s2.split())
  try:
    return len(A.intersection(B))/func(len(A),len(B))
  except ZeroDivisionError:
    return 0
