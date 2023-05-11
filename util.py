from math import sqrt

class Grid:
    facing = [(1,0),(0,1),(-1,0),(0,-1)]
    def __init__(self, **args):
        if 'content' in args:
            self.grid = [[*line] for line in args['content']]
        else:
            self.grid = []
            for _ in range(args['height']):
                self.filler = args['fill'] if 'fill' in args else '_'
                self.grid.append([self.filler for _ in range(args['width'])])
    
    def get(self, *args):
        x,y = args[0][:2] if type(args[0]) is tuple else args
        if y>= 0 and y<len(self.grid) and x>=0 and x<len(self.grid[y]):
            return self.grid[y][x]
        
    def step(self,pos):
        x,y,d = pos
        xOff,yOff = self.facing[d]
        return ((x+xOff)%len(self.grid[0]), (y+yOff)%len(self.grid), d)
    
    def next(self,pos):
        pos = self.step(pos)
        while self.get(pos) == ' ':
            pos = self.step(pos)
        return pos
        
    def filled(self, *args):
        val = self.get(args)
        return True if val is not None and val != self.filler else False
        
    
    def set(self, *args, **kwargs):
        x,y,val = [*args[0], args[1]] if type(args[0]) is tuple else args
        self.grid[y][x] = val
    
    def find(self, target):
        for y in range(len(self.grid)):
            line = self.grid[y]
            for x in range(len(line)):
                if callable(target):
                    if target(line[x]):
                        return (x,y)
                else:
                    if target == line[x]:
                        return (x,y)
    
    def spread(self, *args):
        x,y,diag = [*args[0],args[1]] if type(args[0]) is tuple else args
        result = set()
        for yOff in [-1,0,1]:
            newY = y + yOff
            if newY >= 0 and newY < len(self.grid):
                for xOff in [-1,0,1]:
                    if (xOff != 0 or yOff != 0) and (diag or yOff == 0 or xOff == 0):
                        newX = x + xOff
                        if newX >= 0 and newX < len(self.grid[newY]):
                            result.add((newX, newY))
        return result
    
    def fill(self, start, finish, filler='#'):
        for x in range(min(start[0],finish[0]), max(start[0],finish[0])+1):
            for y in range(min(start[1],finish[1]), max(start[1],finish[1])+1):
                self.set(x, y, filler)
    
    def coordinates(self):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                yield (x,y)
    
    def __str__(self):
        return '\n'.join([''.join(line) for line in self.grid])

stupidNames = [1,2,3,4,5,6]
class CubeFace:
    def __init__(self, x, y):
        self.name = str(stupidNames.pop(0))
        self.x = x
        self.y = y
        self.rotation = [0]*4
        self.neighbor = [None]*4
    
    def put(self, face, pos, rot=0):
        p = pos%4
        if not self.neighbor[p]:
            self.neighbor[p] = face
            self.rotation[p] = rot%4
    
    def unfinished(self):
        for n in self.neighbor:
            if not n:
                return True
    
    def fold(self):
        for i in range(4):
            c = self.neighbor[i]
            n = self.neighbor[(i+1)%4]
            rc = self.rotation[i]
            rn = self.rotation[(i+1)%4]
            if c and n:
                n.put(c, i-rn, rc+1-rn)
                c.put(n, i+1-rc, rn-1-rc)
    
    def getAbsolute(self, fx, fy, d, rot, a):
        for _ in range((4-rot)%4):
            d = d+1
            tx = fx
            fx = a-fy-1
            fy = tx
        return (self.x*a+fx,self.y*a+fy,d%4)
    
class Cube:
    def __init__(self, grid):
        self.grid = grid
        self.a = int(sqrt(sum(len([c for c in l if c != ' ']) for l in grid.grid)//6))
        normalized = Grid(content=[line[::self.a] for line in grid.grid[::self.a]])
        self.faces = dict()
        for x,y in normalized.coordinates():
            if normalized.get(x,y) != ' ':
                self.faces[(x,y)] = (face := CubeFace(x,y))
                for d in range(4):
                    pos = (x+grid.facing[d][0],y+grid.facing[d][1])
                    if pos in self.faces:
                        neighbor = self.faces[pos]
                        face.put(neighbor, d)
                        neighbor.put(face, d+2)
        faces = self.faces.values()
        while len([f for f in faces if f.unfinished()]) > 0:
            for face in self.faces.values():
                face.fold()
    
    def get(self, *args):
        return self.grid.get(*args)
        
    def next(self,pos):
        nextPos = self.grid.step(pos)
        if pos[0]//self.a != nextPos[0]//self.a or pos[1]//self.a != nextPos[1]//self.a:
            face = self.faces[(pos[0]//self.a,pos[1]//self.a)]
            n = face.neighbor[pos[2]]
            nextPos = n.getAbsolute(nextPos[0]%self.a,nextPos[1]%self.a,nextPos[2],face.rotation[pos[2]],self.a)
        return nextPos
