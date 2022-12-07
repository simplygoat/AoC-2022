def group(l, last_index=-1):
    return [l[last_index+1:(last_index:=index)] for index in range(len(l)+1) if index>=len(l) or len(l[index]) == 0]

def day_1(content):
    calories = [sum([int(string) for string in entry]) for entry in group(content)]
    print(max(calories))
    calories.sort()
    print(sum(calories[-3:]))
    
def day_2(content, beats={'A':'B','B':'C','C':'A'}):
    num = lambda n,b='A': ord(n)-ord(b)+1
    calc = lambda l,r: (6 if num(beats[l])==num(r,'X') else (3 if num(l)==num(r,'X') else 0)) + num(r,'X')
    print(sum([calc(*entry.split()) for entry in content]))
    calc2 = lambda l,r: (6+num(beats[l]) if r=='Z' else (3+num(l) if r=='Y' else 0+num(list(beats.keys())[list(beats.values()).index(l)])))
    print(sum([calc2(*entry.split()) for entry in content]))
    
def day_3(content):
    num = lambda n: 1+ord(n)-ord('a') if ord(n) >= ord('a') else 27+ord(n)-ord('A')
    print(sum([num({*line[:int(len(line)/2)]}.intersection({*line[int(len(line)/2):]}).pop()) for line in content]))
    badge = lambda a,b,c:{*a}.intersection({*b}).intersection({*c}).pop()
    print(sum([num(badge(*content[step*3:(step+1)*3])) for step in range(int(len(content)/3))]))

def day_4(content):
    split = [[part.split('-') for part in line.split(',')] for line in content]
    contained = lambda a,b: int(a[0])>=int(b[0]) and int(a[1])<=int(b[1])
    print(len([pair for pair in split if contained(*pair) or contained(*pair[::-1])]))
    overlapped = lambda a,b: int(a[0])<=int(b[1]) and int(a[1])>=int(b[0])
    print(len([pair for pair in split if overlapped(*pair) or overlapped(*pair[::-1])]))
    
def day_5_crane(crates, commands, simple):
    target = [[] for _ in range(int(len(crates[0])/4)+1)]
    for row in [[*line[1::4]] for line in crates[:-1]]:
        for index in range(len(row)):
            if row[index] != ' ':
                target[index].append(row[index])
    for a,f,t in [[*line.split()][1::2] for line in commands]:
        boxes = target[int(f)-1][:int(a)][::-1] if simple else target[int(f)-1][:int(a)]
        target[int(t)-1]=boxes+target[int(t)-1]
        target[int(f)-1]=target[int(f)-1][int(a):]
    return ''.join([crate[0] for crate in target])
    
def day_5(content):
    print(day_5_crane(*group(content), True))
    print(day_5_crane(*group(content), False))
    
def day_6(content):
    find = lambda s,l: [len({*s[index:index+l]}) for index in range(len(s)-l)].index(l)+l
    print(find(content[0], 4))
    print(find(content[0], 14))

def day_7(content, path = [], sizes = {}):
    for parts in [line.split(' ') for line in content[1:]]:
        if parts[0] == '$' and parts[1] == 'cd':
            path = (path + [parts[2]] if parts[2] != '..' else path[:-1])
        elif parts[0].isnumeric():
            for fullPath in ['/'+'/'.join(path[:i]) for i in range(len(path)+1)]:
                sizes.update({fullPath: int(parts[0]) if sizes.get(fullPath) is None else sizes[fullPath] + int(parts[0])})
    print(sum([size for size in sizes.values() if size <= 100000]))
    print(min([size for size in sizes.values() if size > sizes['/']-40000000]))
    
