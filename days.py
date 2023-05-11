from util import Grid,Cube
from functools import reduce
import itertools
import time
import re

def group(content, last_index=-1):
    return [content[last_index+1:(last_index:=index)] for index in range(len(content)+1) if index>=len(content) or len(content[index]) == 0]

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
    
def day_10(content):
    ram = dict(r = 1, c = 1, p = [], s=[])
    cycle = lambda r: dict(r=r['r'], c=r['c']+1, p=r['p']+[r['r']] if r['c']%40==20 else r['p'], s=r['s']+['.' if abs((r['c']-1)%40-r['r'])>1 else '#'])
    for instruction in content:
        ram = cycle(ram)
        if instruction.startswith('addx'):
            ram = cycle(ram)
            ram['r'] += int(instruction[5:])
    print(sum([v*c for v,c in [(ram['p'][i],20+i*40)for i in range(len(ram['p']))]]))
    print('\n'.join([''.join(ram['s'][i*40:(i+1)*40]) for i in range(int(len(ram['s'])/40))]))
    
def day_11(content):
    monkeys = [[[int(w) for w in i[18:].split(', ')],o[22:],int(t[21:]),int(y[29:]),int(n[30:]),0] for _,i,o,t,y,n in group(content)]
    init = lambda number,divs: [(d,number%d) for d in divs]
    apply = lambda worry,operation: [(key,eval('old'+operation)%key) for key,old in worry]
    for monkey in monkeys:
        monkey[0] = [init(item, [m[2] for m in monkeys]) for item in monkey[0]]
    for _ in range(10000):
        for monkey,worry,operation,test,yes,no,_ in [[m,*m] for m in monkeys]:
            for smart in [apply(worry.pop(0), operation) for _ in range(len(worry))]:
                monkey[5] += 1
                monkeys[yes if [value for key,value in smart if key == test][0] == 0 else no][0].append(smart)
    active = [m[5] for m in monkeys]
    active.sort()
    print(active[-1]*active[-2])
    
def day_12_run(grid, start, goal, heightCompare):
    dead = set()
    live = {grid.find(start)}
    height = lambda x: ord('a') if x == 'S' else ord('z') if x == 'E' else ord(x)
    for steps in itertools.count(start=1):
        dead.update(live)
        nextLive = set()
        for xyTuple in live:
            for spread in [s for s in grid.spread(xyTuple, False) if s not in dead and s not in nextLive and heightCompare(height(grid.get(xyTuple)), height(grid.get(s)))]:
                nextLive.add(spread)
                if grid.get(spread) == goal:
                    return steps
        live = nextLive
    
def day_12(content):
    grid = Grid(content=content)
    print(day_12_run(grid,'S','E',lambda ha,hb: hb-ha<=1))
    print(day_12_run(grid,'E','a',lambda ha,hb: ha-hb<=1))

def day_14_sand(grid, height, xOffset, hasFloor, x=500, y=0):
    x-=xOffset
    for y in range(height+1):
        if grid.filled(x,y) and grid.filled(x:=x-1,y) and grid.filled(x:=x+2,y):
            grid.set(x-1,y-1,'O')
            return y-1
    if hasFloor:
        grid.set(x,height-1,'O')
        return height-1

def day_14_simul(walls, hasFloor=False):
    lim = lambda f,i,w: f([f([s[i] for s in d]) for d in w])
    height = lim(max,1,walls)+2
    xOff, xMax = [500-height,500+height] if hasFloor else [lim(min,0,walls),lim(max,0,walls)]
    grid = Grid(width=(xMax-xOff)+1, height=height)
    for dir in walls:
        for xf,yf,xt,yt in [[*dir[i],*dir[i+1]] for i in range(len(dir) - 1)]:
            grid.fill((xf-xOff,yf),(xt-xOff,yt))
    for steps in itertools.count(start=1 if hasFloor else 0):
        rest = day_14_sand(grid, height, xOff, hasFloor)
        if rest is None or rest == 0:
            return steps
    print(grid)

def day_14(content):
    walls = [[(int(pair.split(',')[0]),int(pair.split(',')[1])) for pair in line.split(' -> ')] for line in content]
    print(day_14_simul(walls))
    print(day_14_simul(walls, True))
    
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
            
def day_18(content):
    adj = lambda drop: [(*[(drop[c] if p%3!=c else drop[c]+(1 if p<3 else -1)) for c in range(3)],) for p in range(6)]
    drops = set([(*[int(n) for n in line.split(',')],) for line in content])
    size = 0
    for drop in drops:
        for p in adj(drop):
            size += 0 if p in drops else 1
    print(size)
    negative = set()
    for x in range(max(d[0] for d in drops)+2):
        for y in range(max(d[1] for d in drops)+2):
            for z in range(max(d[2] for d in drops)+2):
                negative.add((x,y,z))
    negative = negative.difference(drops)
    active = {(0,0,0)}
    while len(active) > 0:
        negative = negative.difference(active)
        newActive = set()
        for drop in active:
            for p in adj(drop):
                if p in negative:
                    newActive.add(p)
        active = newActive
    drops = drops.union(negative)
    size = 0
    for drop in drops:
        for p in adj(drop):
            size += 0 if p in drops else 1
    print(size)

def day_19_simul(cost, lim=24, m=100):
    add = lambda a,b,t=1: [a[i]+(b[i]*t) for i in range(len(a))]
    sub = lambda a,b: [a[i]-b[i] for i in range(len(a))]
    affordable = lambda a,b: min(a[i]-b[i] for i in range(len(a)))>=0
    t = [max(c[0] for c in cost),cost[2][1],cost[3][2]]
    states = [([1,0,0,0],[0,0,0,0])]
    for _ in range(lim):
        newStates = []
        states = sorted(states, key = lambda s: add(*s)[::-1], reverse=True)[:m] if len(states) > m else states
        for r,o in states:
            if affordable(o,cost[3]):
                newStates.append(([r[:3]+[r[3]+1],sub(add(o,r),cost[3])]))
            else:
                newStates.append(([*r],add(o,r)))
                for i in range(4):
                    if (i==3 or r[i]<t[i]) and affordable(o,cost[i]):
                        newStates.append(([[r[n]+1 if n==i else r[n] for n in range(4)],sub(add(o,r),cost[i])]))
        states = newStates
    return max(s[1][3] for s in states)
    
def day_19(content):
    costs = []
    for i in range(len(content)):
        c = [[int(s[:s.index(' ')]) for s in cost[(cost.index('costs')+6):].split(' and ')] for cost in content[i].split('. ')]
        costs.append([[*c[0],0,0,0],[*c[1],0,0,0],[*c[2],0,0],[c[3][0],0,c[3][1],0]])
    print(reduce(lambda a,b:a+b, map(lambda i: day_19_simul(costs[i])*(i+1),[i for i in range(len(costs))])))
    print(reduce(lambda a,b:a*b, map(lambda i: day_19_simul(costs[i],32),[i for i in range(3)])))

def day_20_mix(content, key=1, times=1):
    index = lambda l,v,p=0: next(i for i in range(len(l)) if t[i][p]==v)
    s = [(i, int(content[i])*key) for i in range(len(content))]
    t = [*s]
    for _ in range(times):
        for n in s:
            fromIndex = index(t,n[0])
            t.remove(n)
            t.insert((fromIndex + n[1])%len(t), n)
    return sum(t[(index(t,0,1)+o)%len(t)][1] for o in ((i+1)*1000 for i in range(3)))

def day_20(content):
    print(day_20_mix(content))
    print(day_20_mix(content, 811589153, 10))

def day_21(content,m=dict(),o=['*','/','+','-'],r=[lambda v,t,_:t//v,lambda v,t,r:v//t if r else t*v,lambda v,t,_:t-v,lambda v,t,r:v-t if r else t+v]):
    solve = lambda s,p,n,i=0: s(s,p[:(i:=len(p)-p[::-1].index(')'))], r[o.index(p[i])](int(p[i+1:]),n,False)) if (p:=p[1:-1])[0]=='(' else s(s,p[(i:=p.index('(')):], r[o.index(p[i-1])](int(p[:i-1]),n,True)) if p[-1]==')' else n
    handle = lambda a,b,k,v: ((solve(solve,a,b) if type(a) is str else solve(solve,b,a)) if k=='root' else f'({a}{v}{b})') if type(a) is str or type(b) is str else eval(f'int(a{v}b)')
    get = lambda s,ms,m,adv=False: '(x)' if adv and m=='humn' else ms[m] if type(ms[m]) is int else handle(s(s,ms,ms[m][0],adv),s(s,ms,ms[m][2],adv),m,ms[m][1])
    for monkey, action in (l.split(': ') for l in content):
        m[monkey]= int(action) if action.isnumeric() else action.split(' ')
    print(get(get,m.copy(),'root'))
    print(get(get,m,'root',True))

def day_22_step(grid,pos,distance):
    for _ in range(distance):
        nPos = grid.next(pos)
        if grid.get(nPos) == '#':
            #print('bonk')
            return pos
        pos = nPos
    #print('went the distance')
    return pos

def day_22_run(grid,commands,pos):
    for i in commands:
        pos = day_22_step(grid, pos, int(i)) if i.isnumeric() else (*pos[:2],(pos[2]+(1 if i=='R' else -1))%4)
    return (1000*(pos[1]+1))+(4*(pos[0]+1))+pos[2]

def day_22(content):
    width = max(len(l) for l in content[:-2])
    grid = Grid(content=[l.ljust(width) for l in content[:-2]])
    commands = re.findall(r"\d+|[LR]", content[-1])
    print(day_22_run(grid, commands, grid.next((0,0,0))))
    print(day_22_run(Cube(grid), commands, grid.next((0,0,0))))
    
def day_23_print(pos):
    ly = min(y for _,y in pos)
    hy = max(y for _,y in pos)+1
    lx = min(x for x,_ in pos)
    hx = max(x for x,_ in pos)+1
    for y in range(ly,hy):
        print(''.join(['#' if (x,y) in pos else '.' for x in range(lx,hx)]))

def day_23_move(initial,limit):
    pos = initial.copy()
    for i in itertools.count():
        move = dict()
        for p in pos:
            n = [0]*4
            for xOff in range(-1,2):
                for yOff in range(-1,2):
                    if (xOff != 0 or yOff != 0) and (p[0]+xOff,p[1]+yOff) in pos:
                        if yOff == -1:
                            n[0] = n[0]+1
                        elif yOff == 1:
                            n[1] = n[1]+1
                        if xOff == -1:
                            n[2] = n[2]+1
                        elif xOff == 1:
                            n[3] = n[3]+1
            if sum(n) > 0:
                for d in range(4):
                    pref = (d+i)%4
                    if n[pref] == 0:
                        t = (p[0],p[1]+(2*pref-1)) if pref < 2 else (p[0]+(2*(pref-2)-1),p[1])
                        move[t] = None if t in move else p
                        break
        if len(move.keys()) == 0:
            return i+1
        for val in move.items():
            if val[1]:
                pos.remove(val[1])
                pos.add(val[0])
        if limit > 0 and (i+1) >= limit:
            break
    h = max(y for _,y in pos)-min(y for _,y in pos)+1
    w = max(x for x,_ in pos)-min(x for x,_ in pos)+1
    return h*w-len(pos) 

def day_23(content):
    pos = set()
    grid = Grid(content=content)
    for x,y in grid.coordinates():
        if grid.get(x,y) == '#':
            pos.add((x,y))
    print(day_23_move(pos,10))
    print(day_23_move(pos,-1))

def day_24_route(fromPos,toPos,l,r,u,d,off=1):
    wind = lambda x,y,o: (x-o)%len(u) in r[y] or (x+o)%len(u) in l[y] or (y-o)%len(r) in d[x] or (y+o)%len(r) in u[x]
    pos=set()
    for m in itertools.count(start=off):
        newPos = set()
        for x,y in pos:
            if x==toPos[0] and y==toPos[1]:
                return m
            # wait
            if not wind(x,y,m):
                newPos.add((x,y))
            # left
            if x>0 and not wind(x-1,y,m):
                newPos.add((x-1,y))
            # right
            if x<len(u)-1 and not wind(x+1,y,m):
                newPos.add((x+1,y))
            # up
            if y>0 and not wind(x,y-1,m):
                newPos.add((x,y-1))
            # down
            if y<len(r)-1 and not wind(x,y+1,m):
                newPos.add((x,y+1))
        if not wind(*fromPos,m):
            newPos.add(fromPos)
        pos = newPos
        #print(m,pos)

def day_24(content):
    start = [l[1:-1] for l in content[1:-1]]
    l,r,u,d = [[],[],[],[]]
    for y in range(len(start)):
        r.append(set())
        l.append(set())
        for x in range(len(start[y])):
            if y == 0:
                u.append(set())
                d.append(set())
            val = start[y][x]
            if val == '>':
                r[y].add(x)
            if val == '<':
                l[y].add(x)
            if val == 'v':
                d[x].add(y)
            if val == '^':
                u[x].add(y)
    entryPos = (0,0)
    exitPos = (len(u)-1,len(r)-1)
    first = day_24_route(entryPos,exitPos,l,r,u,d)
    print(first)
    back = day_24_route(exitPos,entryPos,l,r,u,d,first)
    print(day_24_route(entryPos,exitPos,l,r,u,d,back))

def day_25_add(a,b,v=['=','-','0','1','2'],r=0,s=''):
    for i in range(max(len(a),len(b))):
        t = r+(0 if i>=len(a) else v.index(a[len(a)-i-1])-2)+(0 if i>=len(b) else v.index(b[len(b)-i-1])-2)
        r = -1 if (t<-2 and (t:=t+5)>=-2) else 1 if (t>2 and (t:=t-5)<=2) else 0
        s = v[t+2]+s
    return s if r == 0 else v[r+2]+s

def day_25(content):
    print(reduce(lambda a,b: day_25_add(a,b), content))
