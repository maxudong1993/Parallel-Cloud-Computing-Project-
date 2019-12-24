import json
from collections import Counter
from mpi4py import MPI
import re
import os

GRID_FILE = "./melbGrid.json"
TWIT_FILE = "./bigTwitter.json"
HASHTAG_EXP = re.compile('(?<= )#\S+?(?= )')  #used to match hashtags

# This class defines information of a grid cell of an area
class Grid_cell:
    def __init__(self, id, boundary):
        self.id = id
        self.xmin, self.xmax, self.ymin, self.ymax = boundary
        self.count = 0
        self.hashtags_freq = Counter()

    # return id of this area (e.g. 'A1','C2','D3')
    def get_id(self):
        return self.id

    # return the number of twitter posts in this area
    def get_count(self):
        return self.count

    # return a counter of all hashtags in this area
    def get_hashtags_freq(self):
        return self.hashtags_freq

    # judge whether the post geo is in this area's range
    def is_include(self, postgeo):
        if postgeo == []:
            return False
        elif postgeo[0] >= self.xmin and postgeo[0] <= self.xmax and postgeo[1] >= self.ymin and postgeo[1] <= self.ymax:
            return True
        else:
            return False

    # add 1 to post count in this area
    def add_count(self):
        self.count += 1

    # add the count of this hashtag in this area
    def add_hashtag(self, hashtag):
        self.hashtags_freq[hashtag] += 1

# load grid file and create instances of Grid_cell class for each area based on 
# their coordinates ranges, return the list of these instances
def read_grid_file(grid_file):
    gridcells = []
    with open(grid_file, "r", encoding='utf-8') as f1:
        js = json.load(f1)
    for ft in js['features']:
        gridcells.append(Grid_cell(ft["properties"]['id'], (ft["properties"]['xmin'],ft["properties"]['xmax'],ft["properties"]['ymin'],ft["properties"]['ymax'])))
    return gridcells

# handle the json line, 
def handle_line(jsline, grid_cells):
    try:
        xy = jsline['doc']['coordinates']['coordinates']
        hashtags = set(HASHTAG_EXP.findall(jsline['doc']['text'].lower()))
        for cell in grid_cells:
            if cell.is_include(xy):
                cell.add_count()
                for hashtag in hashtags:
                    cell.add_hashtag(hashtag)
                break
    except:
        pass

# return the offsets that divides file into 'size' parts
def file_divider(file_path, size):
    file_size = os.path.getsize(file_path)
    divided_size = file_size // size
    offsets = []
    for i in range(size):
        offsets.append(i * divided_size)
    offsets.append(file_size)
    return offsets

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  #number of total processes
    rank = comm.Get_rank()  #id of current processes

    grid_cells = read_grid_file(GRID_FILE)  #list that save instances of all areas

    offsets = file_divider(TWIT_FILE, size)
    with open(TWIT_FILE, "r", encoding='utf-8') as f2:
        offset = offsets[rank]  #location where the process start to read
        stop = offsets[rank + 1]  #location where the process stop reading
        f2.seek(offset)  #move pointer to this location
        while offset < stop:
            line = f2.readline().strip('\n ,')
            offset = f2.tell()  #update offset after each reading
            if line.startswith('{') and line.endswith('}'):
                try:
                    line = json.loads(line)
                    handle_line(line,grid_cells)
                except:
                    pass

    data = {}
    for cell in grid_cells:
        data[cell.get_id()] = (cell.get_count(),cell.get_hashtags_freq())
    comm.Barrier()
    gathered_data = comm.gather(data, root=0) #gather data from other processes to process 0


    if rank == 0:
        # sum up required data from gathered data
        count_ranks = Counter()
        freq_ranks = {}
        for _id in data:
            count_ranks[_id] = sum([d[_id][0] for d in gathered_data])
            freq_ranks[_id] = Counter()
            for d in gathered_data:
                freq_ranks[_id] += d[_id][1]

        # print the ranks of all areas' post number
        print("-------Counts of twitters in each grid cell-------")
        count_ranks = count_ranks.most_common()
        for item in count_ranks:
            print('{}:{} posts'.format(item[0],item[1]))
        print()

        # print top5 frequent hashtags in each areas (solve ties)
        print("-------The top 5 hashtags in each grid cell-------")
        for item in count_ranks:
            print(item[0], end=':(')
            hashtag_ranks = freq_ranks[item[0]].most_common()
            hashtag_topN = []
            N = 5
            pre = 0
            for item in hashtag_ranks:
                if item[1] != pre:
                    N -= 1
                    pre = item[1]
                if N < 0:
                    break
                hashtag_topN.append(item)
            length = len(hashtag_topN)
            for i,hashtag_count in enumerate(hashtag_topN):
                if i < length - 1:
                    print('({},{}),'.format(hashtag_count[0],hashtag_count[1]),end = '')
                else:
                    print('({},{})'.format(hashtag_count[0],hashtag_count[1]),end = '')
            print(')')

if __name__ == '__main__':
    main()
