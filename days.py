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
                sizes[fullPath] = sizes.get(fullPath, 0) + int(parts[0])
    print(sum([size for size in sizes.values() if size <= 100000]))
    print(min([size for size in sizes.values() if size > sizes['/']-40000000]))
    
def day_9_rope(content, length=2, direction=['U','R','D','L'], diff=[(0,-1),(1,0),(0,1),(-1,0)]):
    rope, visited = [[(0,0) for _ in range(length)], set()]
    norm = lambda n: 0 if n == 0 else int(n/abs(n))
    follow = lambda f,b: (b[0]+norm(f[0]-b[0]), b[1]+norm(f[1]-b[1])) if abs(f[0]-b[0]) > 1 or abs(f[1]-b[1]) > 1 else b
    for moveDir, moveCount in [line.split(' ') for line in content]:
        for md in [diff[direction.index(moveDir)] for _ in range(int(moveCount))]:
            for i in range(len(rope)):
                rope[i] = (rope[0][0]+md[0], rope[0][1]+md[1]) if i == 0 else follow(rope[i-1], rope[i])
            visited.add(','.join([str(i) for i in rope[len(rope)-1]]))
    return len(visited)

def day_9(content):
    print(day_9_rope(content))
    print(day_9_rope(content, 10))
    
 def day_17(content, p=[[(0,4)],[(1,2),(0,3),(1,2)],[(0,3),(2,3),(2,3)],[(0,1),(0,1),(0,1),(0,1)],[(0,2),(0,2)]], ls=0, i=-1, grid=[], ih=0, ib=[], sb=[], t=1000000000000):
    mayPut = lambda g,p,xO,yO: sum([len([x for x,y in _ if g[y+yO][x+xO]=='#']) for _ in [[[x,y] for x in range(*p[y])] for y in range(min(len(g)-yO,len(p)))]])==0
    for step in itertools.cycle([0,1,2,3,4]):
        if i>=0 and ((ih:=ih+i)>=0 and step == 0):
            if ih in ib and (ih:=ib.index(ih)) >= 0:
                print(sum(sb[:ih])+(sum(sb[ih:])*((t-ih*5)//(len(sb[ih:])*5)))+sum(sb[ih:][:((t-ih*5)%(len(sb[ih:])*5))//5]))
                return
            ih,ls,_,_ = [0, len(grid), ib.append(ih), sb.append(len(grid)-ls)]
        piece, width, x, ih = [p[step], max([t[1] for t in p[step]]), 2, 0 if step == 0 else ih*len(content[0])]
        for y in range(len(grid)+3,-2,-1):
            if y==-1 or (y<len(grid) and not mayPut(grid, piece, x, y)):
                grid.extend([['.']*7 for _ in range(max(0,(y+1+len(piece))-len(grid)))])
                for yOff in range(len(piece)):
                    for xOff in range(*piece[yOff]):
                        grid[y+1+yOff][x+xOff] = '#'
                break
            xOff = -1 if content[0][(i:=i+1 if i+1<len(content[0]) else 0)]=='<' else 1
            x = (x+xOff) if (x+xOff)>=0 and (x+xOff+width)<=7 and (y>=len(grid) or mayPut(grid,piece,x+xOff,y)) else x
