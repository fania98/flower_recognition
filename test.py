def thetup():
    for i in range(1,5):
        yield i,i+1

gotup = thetup()
for dd in gotup:
    (j, f)=dd
    print(j)
    print(f)