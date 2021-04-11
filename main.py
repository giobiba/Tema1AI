from collections import Counter
import time
import copy 
import sys
import os
import stopit

class NodParcurgere:
    """
        clasa NodParcurege este folosita pentru a construi arborele
        membrele statice (nr_noduri, max_noduri, total_noduri) sunt folosite pentru a contoriza complexitatea algoritmilor,
        tinand cont de cate noduri sunt create si cate noduri exista in memorie la un mom dat
    """
    nr_noduri = 0
    max_noduri = 0
    total_noduri = 0

    def __init__(self, info, parent, cost = 0, h = 0):
        self.info = info
        self.parent = parent
        self.g = cost
        self.h = h
        self.f = self.g + self.h
        self.__class__.nr_noduri += 1
        self.__class__.total_noduri += 1
        self.__class__.max_noduri = max(self.__class__.max_noduri, self.__class__.nr_noduri)

    @classmethod
    def reset(cls):
        """
            Functie utilitara care ne ajuta la contorizarea numarului maxim de noduri in memorie respectiv numarului total de noduri calculate
        """
        cls.total_noduri = 0
        cls.max_noduri = 0
        cls.nr_noduri = 0

    def getPath(self):
        """
            Returneaza drumul de la nodul curent la radacina
        """
        l = [self]
        nod = self
        while nod.parent is not None:
            l.insert(0, nod.parent)
            nod = nod.parent
        return l

    def showPath(self, file_output = sys.stdout):
        l = self.getPath()
        print(l[0], file = file_output)
        prev = l[0].g
        for nod in l[1:]:
            print(str(nod), file = file_output)
            print("Cost mutare: {},".format(round(nod.g - prev, 4)), file = file_output)
            prev = nod.g
        print("S-au realizat {} mutari cu costul {}.".format(len(l), round(self.g, 4)), file = file_output)
        print("Numarul maxim de noduri in memorie: ", self.__class__.max_noduri, file = file_output)
        print("Numarul total de noduri calculate: ", self.__class__.total_noduri, file = file_output)

        return len(l)
    
    def existsInPath(self, infoNodNou):
        nodDrum = self
        while nodDrum is not None:
            if infoNodNou == nodDrum.info:
                return True
            nodDrum = nodDrum.parent

        return False

    def __repr__(self):
        sir = ""
        for linie in self.info:
            sir += " " + linie + "\n"
        return sir

    def __str__(self):
        sir = ""
        for linie in self.info:
            sir += linie + "\n"
        return sir

    def __del__(self):
        self.__class__.nr_noduri = max(self.__class__.nr_noduri - 1, 0)

class Graph:
    heuristics = ["ordinary heuristic", "heuristic 1", "heuristic 2", "bad heuristic"]
    
    def __init__(self, fisier_input):
        f = open(fisier_input, "r")
        content =  f.read().split()
        self.start  = list(map(lambda x: x.strip(), content))

        height = len(self.start)
        width = len(self.start[0])

        self.scopes = [['#' * width for i in range(height)]]
  
    def test_scope(self, curr):
        return curr.info in self.scopes

    def get_alphabet(self, infoNod):
        """
            returneaza multimea de "culori" prezente in starea
        """
        listaInfo = filter(lambda ch: ch != '#', "".join(infoNod))
        frequency = Counter(listaInfo)
        
        return frequency

    def verifyExistance(self, infoNod): 
        """
            Aceasta functie verifica daca starea in care ne aflam nu va ajunge niciodata la o stare finala, numarand cate patrate de o anumita culoare exista. Daca acel numar este < 3 este logic ca nu vor putea fi eliminate niciodata
        """

        frequency = self.get_alphabet(infoNod)

        for k, v in frequency.items():
            if v < 3:
                return False
        return True 

    def exploreBlock(self, infoNod, i, j):
        """
            Aceasta functie este folosita pentru a delimita blockuri de culori dintr-o stare
            Dandu-se un i si un j se exploreaza tabloul specific starii si se returneaza o lista de tupluri care reprezinta coordonatele fiecarui patrat din block
            infoNod este modificat astfel incat blockul pe care l-am explorat sa fie scos din tablou
        """
        
        matrix = list(map(list, infoNod))
        block = [(i, j)]
        # coada pe care o folosim in explorarea tabloului
        q = [(i, j)]
        # retinem culoarea blockului deoarece tabloul va suferi modificari
        color = infoNod[i][j]
        # lista de patrate vizitate
        visited = [(i,j)]

        while q:
            i, j = q.pop()

            matrix[i][j] = '#'

            # pentru toate cele 4 patrate din vecinatatea noastra
            # verificam daca suntem inca in tablou 
            # daca da adaugam patratul in coada in cazul in care nu l-am mai vizitat si are aceeasi culoare cu patratul initial
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                i1 = i + dx
                j1 = j + dy
                
                if i1 < 0 or i1 > len(matrix) - 1 or j1 < 0 or j1 > len(matrix[i]) - 1:
                    continue
                
                if matrix[i1][j1] == color and (i1, j1) not in visited:
                    visited.append((i1,j1))
                    q.append((i1, j1))
                    block.append((i1, j1))     
        infoNod = list(map(lambda ls: "".join(ls), matrix)) 
        return infoNod, block

    def splitIntoBlocks(self, infoNod):
        """
            In aceasta functie apelam recursiv exploreBlock() cat timp mai exista block-uri in tabloul nostru
            Returnam toate blockurile, acestea desemnand fiecare posibila eliminiare (mutare) din starea curenta
        """
        blocks = []

        for i in range(len(infoNod)):
            for j in range(len(infoNod[i])):
                if infoNod[i][j] == '#':
                    continue
                infoNod, block = self.exploreBlock(infoNod, i, j)
                blocks.append(block)
        return blocks

    def count(self, color, matrix):
        """
            Functie utilitara pentru a afla numarul de aparitii a unei culori in starea curenta
            Acest lucru este folosit la calcularea costului unei mutari
        """
        counter = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == color:
                    counter += 1
        return counter

    def generateSuccessors(self, curr: NodParcurgere, type_h="ordinary heuristic"):
        """
            Genereaza toate posibilele mutari dintr-o stare si returneaza o lista de noduri cu starile urmatoare
        """
        # pornim de la o lista goala
        successors = []
        infoCopy = copy.deepcopy(curr.info)
        # pentru fiecare posibila eliminare (mutare) calculam starea urmatoare 
        for block in self.splitIntoBlocks(infoCopy):
            if len(block) < 3: # Daca nu exista cel putin 3 patratele in zona pe care dorim sa o eliminam o ignoram
                continue

            cpy = copy.deepcopy(curr.info)

            # Transformam lista de stringuri intr-o lista de lista de charuri pentru a nu mai avea problema cu imutabilitatea stringurilor 
            list_cpy = list(map(list, cpy))

            # Vedem ce culoare are patratele pe care le eliminam
            i, j = block[0]
            color = cpy[i][j]

            # numaram cate patrate de aceasta culoare exista
            N = self.count(color, list_cpy)
            
            # Eliminam blockul de culori selectat
            for i,j in block:
                list_cpy[i][j] = '#'

            # Aplicam gravitatia

            gravity(list_cpy)

            # Dam shift la stanga
            
            shiftLeft(list_cpy)
            
            cpy = list(map(lambda ls: "".join(ls), list_cpy))

            # Daca aceast nod nu va ajunge intr-o stare finala nu il luam in considerare
            if not self.verifyExistance(cpy):
                continue

            # Verificam daca succesorul a mai aparut in path-ul nodului curent
            if curr.existsInPath(cpy):
                continue

            # nr de blockuri eliminate
            K = len(block)

            # calculam costul
            cost = 1 + (N - K) / N
            
            successors.append(
                NodParcurgere(cpy, curr, curr.g + cost, self.calculate_h(cpy, type_h))
            )

        return successors

    def calculate_h(self, infoNod, type_h="ordinary heuristic"):
        if infoNod in self.scopes:
                return 0
        if type_h == "ordinary heuristic":
            return 1
        if type_h == "heuristic 1":
            return self.heuristic1(infoNod)
        if type_h == "heuristic 2":
            return self.heuristic2(infoNod)
        if type_h == "bad heuristic":
            return self.bad_heuristic(infoNod)

    def heuristic1(self, infoNod):
        """
            Motivul pentru care este admisibila este dintr-o stare cu n culori trebuie sa facem cel putin n eliminari ca sa ajungem la o stare finala
        """
        alphabet = self.get_alphabet(infoNod)
        return len(alphabet)

    def heuristic2(self, infoNod):
        """
            O simplificare a euristicii 1, daca avem doar o culoare in starea noastra, inseamna ca dupa urmatoare eliminare suntem intr-o stare finala
            Altfel, o sa necesite cel putin 2 miscari pentru a ajunge la o stare finala
        """
        alphabet = self.get_alphabet(infoNod)
        if len(alphabet) == 1:
            return 1
        else:
            return 2

    def bad_heuristic(self, infoNod):
        """
        In cazul in care am avea stare
        a a a 
        b b b
        a a a
        aceasta euristica ar returna 3, ceea ce ar fi peste distanta reala 2(eliminand prima oara b, apoi a)
        """

        return len(self.splitIntoBlocks(infoNod))

def verifyEmptyColumn(matrix, j):
    """
        Functiei utilitara care verifica daca o anumita coloana este goala
    """
    for row in matrix:
        if row[j] != "#":
            return False
    return True

def shiftLeft(matrix):
    """
        Parcurgem coloana cu coloana, iar in cazul in care gasim una goala dam "shift" la stanga toate coloanele care se afla la dreapta acesteia
    """
    for j in range(len(matrix[0])):
        # In cazul in care coloana nu este empty trecem peste
        if not verifyEmptyColumn(matrix, j):
            continue;
        for i in range(len(matrix)):
            for k in range(j + 1, len(matrix[i])):
                matrix[i][j] = matrix[i][k]
            # ultima coloana o marcam ca goala
            matrix[i][len(matrix[i]) - 1] = '#'

def gravity(matrix):
    """
        Pentru o matrice data dorim sa nu avem # (adica spatii libere) sub o litera a alfabetului, adica
        daca matrix[i][j] = '#' atunci
        si matrix[k][j] = '#' pt k din {0 .. i - 1}
        Parcurgem lista bottom up iar atunci cand gasim o valoare != '#' o coboram cat de jos posibil in coloana in care se afla
    """

    # pornim de la a doua cea mai joasa linie linie in sus 
    for i in range(len(matrix) - 2, -1, -1):
        for j in range(len(matrix[i])):
            # daca este spatiul gol nu luam in considerare
            if matrix[i][j] == '#':
                continue

            # pornim o cautare care ne gaseste ultima aparitia a spatiului gol, unde vom pune caracterul care doream sa l coboram
            k = i + 1
            
            if matrix[k][j] != '#':
                continue
            
            while k < len(matrix) and matrix[k][j] == '#':
                k += 1
            k -= 1

            matrix[k][j] = matrix[i][j]
            matrix[i][j] = '#'
        # print(matrix)

def time_dif(start, end):
    dt = end - start
    return round(dt * 1000, 7)

@stopit.threading_timeoutable(default="S-a depasit limita de timp")
def uniform_cost(gr, nrSolutiiCautate = 1, file_output = sys.stdout, start = time.time()):
    if not gr.verifyExistance(gr.start):
        print("Nu exista solutie", file = file_output)
        return
    
    queue = [NodParcurgere(gr.start, None, 0)]
    nodes_in_memory = 0

    while len(queue) > 0:
        curr = queue.pop(0)

        if gr.test_scope(curr):
            print("Solutie : ", file = file_output)
            curr.showPath(file_output = file_output)
            print("Durata: {} milisecunde".format(time_dif(start, time.time())), file = file_output)
            print("\n----------------\n", file = file_output)
            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                return
        lSuccesori = gr.generateSuccessors(curr)

        # inseram succesorii in coada, tinand cont sa acea coada sa ramana ordonata in functie de g
        for succ in lSuccesori:
            i = 0
            while i < len(queue):
                if queue[i].g > succ.g:
                    break
                i += 1
            queue.insert(i, succ)
    print("Nu exista solutie", file = file_output)

@stopit.threading_timeoutable(default="S-a depasit limita de timp")
def A_star(gr, nrSolutiiCautate = 1, type_h = "ordinary heuristic", file_output = sys.stdout, start = time.time()):
    if not gr.verifyExistance(gr.start):
        print("Nu exista solutie", file = file_output)
        return 
    
    queue = [NodParcurgere(gr.start, None, 0, gr.calculate_h(gr.start, type_h))]
    nodes_in_memory = 0

    while len(queue) > 0:
        curr = queue.pop(0)

        if gr.test_scope(curr):
            print("Solutie : ", file = file_output)
            curr.showPath(file_output = file_output)
            print("Durata: {} milisecunde".format(time_dif(start, time.time())), file = file_output)
            print("\n----------------\n", file = file_output)
            nrSolutiiCautate -= 1
            if nrSolutiiCautate == 0:
                return
        lSuccessors = gr.generateSuccessors(curr, type_h = type_h)

        for succ in lSuccessors:
            i = 0
            while i < len(queue):
                if queue[i].f > succ.f:
                    break
                i += 1
            queue.insert(i, succ)
    print("Nu exista solutie", file = file_output)

@stopit.threading_timeoutable(default="S-a depasit limita de timp")
def A_star_efficient(gr, type_h = "ordinary heuristic", file_output = sys.stdout, start = time.time()):
    if not gr.verifyExistance(gr.start):
        print("Nu exista solutie", file = file_output)
        return

    open = [NodParcurgere(gr.start, None, 0, gr.calculate_h(gr.start ,type_h))]
    closed = []
    nodes_in_memory = 0

    while len(open) > 0:
        curr = open.pop(0)
        closed.append(curr)

        if gr.test_scope(curr):
            print("Solutie : ", file = file_output)
            curr.showPath(file_output = file_output)
            print("Durata: {} milisecunde".format(time_dif(start, time.time())), file = file_output)
            print("\n----------------\n", file = file_output)
            return
        lSuccessors = gr.generateSuccessors(curr, type_h = type_h)
        lSuccessorsCopy = lSuccessors.copy()
        
        for succ in lSuccessorsCopy:
            foundOpen = False
            for elem in open:
                # in cazul in care starea lui succ este in open
                # alegem sa o extindem doar pe cea care area valoarea mai mica a lui f 
                if succ.info == elem.info:
                    foundOpen = True
                    if succ.f < elem.f:
                        open.remove(elem)
                    else:
                        lSuccessors.remove(succ)
                    break
                # in cazul in care nu gasim in open acea stare, verificam daca exista in closed
                # daca acest lucru este adevarat, alegem sa il extindem pe succ doar in cazul in care are un f mai mic decat a elementului prezent in coada
                # de asemenea eliminam din closed acea stare deoarece am gasit una mai buna
            if not foundOpen:
                for elem in closed:
                    if succ.info == elem.info:
                        if succ.f < elem.f:
                            closed.remove(elem)
                        else:
                            lSuccessors.remove(succ)
                        break

        for succ in lSuccessors:
            i = 0
            while i < len(open):
                if open[i].f > succ.f:
                    break
                i += 1
            open.insert(i, succ)
    print("Nu exista solutie", file = file_output)

@stopit.threading_timeoutable(default="S-a depasit limita de timp")
def IDA_star(gr, nrSolutiiCautate = 1, type_h = "ordinary heuristic", file_output = sys.stdout, start = time.time()):
    if not gr.verifyExistance(gr.start):
        print("Nu exista solutie", file = file_output)
        return

    
    startNode = NodParcurgere(
        gr.start, None, 0, gr.calculate_h(gr.start, type_h)
    )
    # setam limita la care vrem sa explorem in urmatoarea iteratie
    limit = startNode.f
    while True:
        nrSolutiiCautate, rez = IDA_iteration(
            gr, startNode, limit, nrSolutiiCautate, type_h, file_output, start = start
        )
        # cazul in care gasim toate solutiile iesim din functie
        if rez == "found":
            return
        # cazul in care noua limita este inf inseamna ca am epuizat toate nodurile si nu am gasit toate solutiile
        if rez == float("inf"):
            print("Nu exista solutie", file = file_output)
            return
        limit = rez

def IDA_iteration(gr, curr, limit, nrSolutiiCautate = 1, type_h= "ordinary heuristic", file_output = sys.stdout, start = time.time()):
    # in cazul in care curr.f > limit, setam noua limita ca fiind curr.f
    if curr.f > limit:
        return nrSolutiiCautate, curr.f

    if gr.test_scope(curr) and curr.f == limit:
        print("Solutie: ", file = file_output)
        curr.showPath(file_output = file_output)
        print("Durata: {} milisecunde".format(time_dif(start, time.time())), file = file_output)
        print("\n----------------\n", file = file_output)
        nrSolutiiCautate -= 1

        if nrSolutiiCautate == 0:
            return nrSolutiiCautate, "found"

    lSuccessors = gr.generateSuccessors(curr, type_h = type_h)
    minim = float("inf")
    
    # apelam functia recursiv pentru toti succesori nodului, verificand la fiecare nod nou sa actualizam minimul pentru a modifica limita din iteratia urmatoare
    for succ in lSuccessors:
        nrSolutiiCautate, rez = IDA_iteration(gr, succ, limit, nrSolutiiCautate, type_h, file_output = file_output, start = start)
        if rez == "found":
            return nrSolutiiCautate, "found"
        if rez < minim:
            minim = rez

    del curr
    return nrSolutiiCautate, minim

if __name__ == "__main__":
    try:
        if len(sys.argv) != 5:
            raise IndexError("Wrong number of arguments")

        # extragem argumentele din apelul programului
        folder_input = sys.argv[1]
        folder_output = sys.argv[2]
        nsol = int(sys.argv[3])
        timeout_time = int(sys.argv[4])
    except IndexError as e:
        # in cazul vreunei greseli programul nu ruleaza
        print(e)
        print("Sablonul pentru a aprela programul este:")
        print("python3 main.py fisier_input fisier_output nrsol timeout")

        exit(-1)
    except:
        print("Bad input")
    if not os.path.exists(folder_input):
        # este necesar ca fisierul de input sa existe
        print("Input folder doesn't exist")
        exit(-1)

    if not os.path.exists(folder_output):
        # creem fisierul de output daca nu exista
        os.mkdir(folder_output)

    # pentru fiecare fisier de input pornim fiecare algoritm cu fiecare euristica in parte
    # scriem in fisierele de output, creeate dupa numele fisierelor de input, rezultatele afisate
    for finput in os.listdir(folder_input):
        gr = Graph(folder_input + '/' + finput)
        
        parse = finput.split('.')
        foutput = folder_output + '/' + parse[0] + "_output." + parse[1]
        
        if not os.path.exists(foutput):
            os.mknod(foutput)

        file_output = open(foutput, 'w')
        
        ###################### UCS #########################
        print("Uniform cost search:", file = file_output)

        start = time.time()

        rez = uniform_cost(gr, nsol, file_output = file_output, start = start, timeout = timeout_time)

        # daca rez nu este None inseamna ca programul a trecut the timeout-ul precizat
        # caz in care scriem si asta in fisier
        if rez is not None:
            print(rez, file = file_output)

        print("############################", file = file_output)
        print('\n', file = file_output)

        NodParcurgere.reset()

        ###################### A* #########################
        print("A*:", file = file_output)

        for type_h in Graph.heuristics:
            print("Heuristic: ", type_h, file = file_output)
            
            start = time.time()

            rez = A_star(gr, nsol, type_h = type_h, file_output = file_output, start = start, timeout = timeout_time)

            if rez is not None:
                print(rez, file = file_output)

            print("############################", file = file_output)
            print('\n', file = file_output)

            NodParcurgere.reset()

        ###################### A* o/c #########################
        print("A* with open/write:", file = file_output)

        for type_h in Graph.heuristics:
            print("Heuristic: ", type_h, file = file_output)

            start = time.time()

            rez = A_star_efficient(gr, type_h = type_h, file_output = file_output, start = start, timeout = timeout_time)

            if rez is not None:
                print(rez, file = file_output)

            print("############################", file = file_output)
            print('\n', file = file_output)

            NodParcurgere.reset()

        ###################### IDA* #########################

        print("IDA*:", file = file_output)

        for type_h in Graph.heuristics:
            print("Heuristic: ", type_h, file = file_output)

            start = time.time()

            rez = IDA_star(gr, nsol, type_h = type_h, file_output = file_output, start = start, timeout = timeout_time)

            if rez is not None:
                print(rez, file = file_output)
            
            print("############################", file = file_output)
            print('\n', file = file_output)