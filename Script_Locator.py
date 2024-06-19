#Author - Zach Marto
#ChatGPT helped too

'''
USER SETTINGS - Manipulate the behavior of the program
'''

# You will need to change these to fit your computer:
# Root directory you will search through to find Python files
search_dir = r"\Users\zachm"
# Directory you will export Python files to
export_dir = r"\Users\zachm\Desktop\Script Locator\Script Locator Indexes"
# Directory to import files to search from
import_dir = r"\Users\zachm\Desktop\Script Locator\Script Locator Indexes\zachm"

# Determine whether you are indexing (slow), searching (fast), or both
index_and_export = False
import_and_search = True

keywords = ["lose connection"]
#Options: 'a' (all), f (functions), 'c' (classes), 'd' (docstrings), 'n' (filename)
search_in = ['a']

#True - exact words in the exact order. Flexible on capitalization,
#spaces, capitalization, and camelCase
#False - Flexible on order. Also works with "similar" words - ie will try to
#match "run" with "running," "ran," "jog," and "sprint"
do_strict_match = False

# True - searches all files/subfolders in search_dir
# False - searches only files in search_dir
index_recursively = True
#Includes python libraries in the files index (can be very expensive)
index_python_libraries = False
#Check if you have the requisite nltk packages installed. Install them
#if not. The check isn't required once you have them installed.
check_nltk_packages = False
#This ensures that a file won't be near the top of the list because
#it's huge and has a few minor matches.
weight_score_by_file_size = True
#How many results are shown after a search. Will only show files that
#have a score greater than 0
num_search_results_shown = 20
#Theoretically this program could work with any file extension that
#can be opned with the python open
file_extension = "py"



'''
PROGRAM CODE
'''

#Lists files in directory that match a pattern
import glob
import time
import datetime
import os.path
import re
#Stores scraped python files externally
try:
    import cPickle as pickle
except:
    import pickle
#Used for sorting match objects
from operator import attrgetter
#Used for links to paths
from pathlib import Path
#NLP library for keyword searching
import nltk
from nltk.stem import WordNetLemmatizer
#Used to split list
# from itertools import islice
# from fuzzywuzzy import fuzz

match_functions = 'f' in search_in or 'a' in search_in
match_classes = 'c' in search_in or 'a' in search_in
match_docstrings = 'd' in search_in or 'a' in search_in
match_name = 'n' in search_in or 'a' in search_in

#Returns a string with only lowercase letters and number
def standardString(s):
    alphanumeric = ''.join([e for e in s if e.isalpha() or e.isdigit()])
    return alphanumeric.lower()

def standardList(old):
    l = old.copy()
    for i in range(0, len(l)):
        l[i] = standardString(l[i])
    return l

def replaceDelimiters(string):
    # Define a regular expression pattern to match all delimiters
    delimiters = re.compile(r"[ _\-.,;!?\"']|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[\n]")
    # Replace all delimiters with a space
    new_string = delimiters.sub(" ", string)
    # Convert all characters to lower case
    return new_string.lower()
    
def lemmatizeTokens(s, lemmatizer):
    # Tokenize the input string
    tokens = nltk.word_tokenize(s)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos = 'v') for token in tokens]
    return lemmatized_tokens

def removeNonAlphaStrings(string_list):
    new_list = []
    for s in string_list:
        alpha_num = False
        for c in s:
            if c.isalpha() or c.isdigit():
                alpha_num = True
                break
        if alpha_num:
            new_list.append(s)
    return new_list

def lemmatizeString(s, lemmatizer):
    # Replace delimiters with space and convert to lower case
    delimited_str = replaceDelimiters(s)
    # Lemmatize the string
    lemmatized_tokens = lemmatizeTokens(delimited_str, lemmatizer)
    return removeNonAlphaStrings(lemmatized_tokens)

def lemmatizeList(l, lemmatizer):
    new_list = []
    for s in l:
        new_list.append(lemmatizeString(s, lemmatizer))
    return new_list

#Stores attributes of a Python script
#pyFiles are created every scrape
class pyFile:
    def __init__(self, path, content, lemmatizer):
        self.path = path
        # self.path_link = self.getPathLink(path)
        self.content = content
        self.name = path.split('\\')[-1][:-3]
        self.functions = self.getFunctionNames()
        self.classes = self.getClassNames()
        self.docstrings = self.getDocStrings()
        
        self.name_standard = standardString(self.name)
        self.functions_standard = standardList(self.functions)
        self.classes_standard = standardList(self.classes)
        self.docstrings_standard = standardList(self.docstrings)
        
        self.lemmatizer = lemmatizer
        self.name_lemmatized = lemmatizeString(self.name, self.lemmatizer)
        self.functions_lemmatized = lemmatizeList(self.functions, self.lemmatizer)
        self.classes_lemmatized = lemmatizeList(self.classes, self.lemmatizer)
        self.docstrings_lemmatized = lemmatizeList(self.docstrings, self.lemmatizer)
    
    def getPathLink(self, p):
        file_path = Path(p)
        link = file_path.as_uri()
        return link
    
    def getStrings(self, start, end, long_string):
        splits = long_string.split(start)
        del splits[0]
        for i in range(0, len(splits)):
            splits[i] = splits[i].partition(end)[0]
        return splits
    
    def getFunctionNames(self):
        return self.getStrings("def ", "(", self.content)
    
    def getClassNames(self):
        return self.getStrings("class ", "(", self.content)

    def getDocStrings(self):
        return re.findall(r"'''(.*?)'''", self.content, re.DOTALL)
    
    
#Contains a Python file and matching score
#Matches are created every search
class match:
    def __init__(self, py_file):
        self.py_file = py_file
        self.score = 0
        self.search_terms = self.concatenateSearchTerms()
    
    def splitLists(self, lists, length):
        new_lists = []
        for l in lists:
            new_lists.append(self.splitList(l, length))
        return new_lists
    
    def splitList(self, l, length):
        c = 0
        split_list = []
        split = []
        for i in l:
            if c == length:
                split_list.append(split)
                c = 0
                split = []
            split.append(i)
            c += 1
        if c > 0:
            split_list.append(split)
        return split_list
    
    def concatenateSearchTerms(self):
        search_terms = []
        
        if do_strict_match:
            if match_functions:
                search_terms += self.py_file.functions_standard
            if match_classes:
                search_terms += self.py_file.classes_standard
            if match_docstrings:
                search_terms += self.py_file.docstrings_standard
            if match_name:
                search_terms += [self.py_file.name_standard]
        else:
            if match_functions:
                search_terms += self.py_file.functions_lemmatized
            if match_classes:
                search_terms += self.py_file.classes_lemmatized
            if match_docstrings:
#                 docstrings = self.splitLists(self.py_file.docstrings_lemmatized, 5)
#                 search_terms += docstrings[0]
                search_terms += self.py_file.docstrings_lemmatized
            if match_name:
                search_terms += [self.py_file.name_lemmatized]
#             for i in range(len(search_terms)):
#                 search_terms[i] = ' '.join(search_terms[i])
        return search_terms

def pyFilesToMatches(py_files):
    matches = []
    for f in py_files:
        matches.append(match(f))
    return matches

#Recursively search a given directory for Python files
def getPythonFileNames(dir):
    if index_recursively:
        dir = dir + "\\**\\" + "*." + file_extension
    else:
        dir = dir + "\\" + "*." + file_extension
    files = glob.glob(dir, recursive = True)
    files.sort()
    return files

def isLibrary(s):
    return re.search(r'python\d+\\Lib\\', s) is not None

def getFileContents(paths):
    contents = []
    new_paths = []
    error_files = []
    for path in paths:
        if not index_python_libraries and isLibrary(path):
            continue
        try:
            with open(path, "r") as file:
                contents.append(file.read())
                new_paths.append(path)
                file.close()
        except Exception as e:
            #Known errors
            #UnicodeDecodeError
            #PermissionError
            error_files.append(path + '   ' + str(e))
    return new_paths, contents, error_files

def getDateTime():
    current_time = str(datetime.datetime.now())
    current_time = current_time.replace(' ', '_')
    current_time = current_time.replace(':', '_')
    current_time = current_time.split('.')[0]
    return current_time

def prepLemmatizer():
    if check_nltk_packages:
        # Download necessary NLTK packages
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('punkt')
    # Create a WordNetLemmatizer object
    return WordNetLemmatizer()

#Take Python file names and turn them into a list
#of pyFiles
def indexPyFiles(lemmatizer):
    print("Beginning Index and Export")
    filePaths = getPythonFileNames(search_dir)
    print("Located all .py files in your directory")
    filePaths, fileContents, errorFiles = getFileContents(filePaths)
    print("Successfully extracted file contents")
    pyFiles = []
    for i in range(0, len(filePaths)):
        pyFiles.append(pyFile(filePaths[i], fileContents[i], lemmatizer))
    return pyFiles, errorFiles

def writeList(f, title, l):
    f.write(title + '\n')
    for i in l:
        s = str(i) + '\n'
        f.write(s)

def prepExportDir():
    global export_dir
    export_dir = export_dir + '\\' + search_dir.split('\\')[-1]
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

#Exports to file stored on server
def exportFiles(py_files, error_files):
    prepExportDir()
    
    #Write files into human-readable output doc which can be searched
    #Manually if necessary
    raw_filename = "Raw_Output_" + getDateTime() + ".txt"
    raw_path = os.path.join(export_dir, raw_filename)
    raw_file = open(raw_path, "w")
    for f in py_files:
        raw_file.write("NAME:: " + f.name + '\n')
        raw_file.write("PATH:: " + f.path + '\n')
        writeList(raw_file, "\nFUNCTIONS::", f.functions)
        writeList(raw_file, "\nCLASSES::", f.classes)
        writeList(raw_file, "\nDOCSTRINGS::", f.docstrings)
#         raw_file.write("CODE::\n\n\n" + f.content + '\n\n\n')
        raw_file.write("\n\n\n")
    raw_file.write("The following files could not be accessed:\n")
    for e in error_files:
        raw_file.write(e + '\n')
    raw_file.close()
    
    #Write files into text file which can be searched by python program
    index_filename = "Indexed_Code_" + getDateTime() + ".txt"
    index_path = os.path.join(export_dir, index_filename)
    index_file = open(index_path, "wb")
    pickle.dump(py_files, index_file)
    index_file.close()
    
    print("Exported " + str(len(py_files)) + " ." + file_extension + " files in %s seconds" % (time.time() - start_time))

#Imports from file stored on server
def importFiles():
    print("Beginning Import and Search")
    files = glob.glob(import_dir + "\\Indexed_Code*.txt")
    
    if len(files) < 1:
        print("No files to import from found in " + import_dir + "\n")
    
    #Takes latest file
    files.sort()
    import_path = files[-1]
    #Convert pickle back to list of pyFiles
    with open(import_path, "rb") as import_file:
        py_files = pickle.loads(import_file.read())
    import_file.close()
    print("Imported " + str(len(py_files)) + " ." + file_extension + " files in %s seconds" % (time.time() - start_time))
    return py_files

#Check if any keywords are in a match
def checkMatch(keyword, match):
    search_terms = match.search_terms
    
    for t in search_terms:
        if do_strict_match:
            if keyword in t:
                match.score += 100
        else:
            found = True
            for k_w in keyword:
                if k_w not in t:
                    found = False
            if found:
                match.score += 100
    
#Sort matches by highest to lowest score
def sortMatches(matches):
    return sorted(matches, key=attrgetter('score'), reverse=True)

def printMatches(matches):
    print("\nResults of searching " + str(len(matches)) + " files:\n")
    for i, m in enumerate(matches):
        if m.score == 0:
            break
        if i >= num_search_results_shown:
            break
        print(str(i+1) + " (" + str(m.score) + ") " + m.py_file.path + '\n')

#Compare pyFiles to keywords and then show results
def searchFiles(py_files, lemmatizer):
    print("Beginning Search")
    matches = pyFilesToMatches(py_files)
    
    for k in keywords:
        if do_strict_match:
            k = standardString(k)
        else:
            k = lemmatizeString(k, lemmatizer)
        for m in matches:
            checkMatch(k, m)
    
    #Adjust score based on content of file
    #Don't return a high score due to a few minor matches
    #over a long file
    if weight_score_by_file_size:
        for m in matches:
            m.score /= len(m.search_terms)
    
    matches = sortMatches(matches)
    printMatches(matches)

start_time = time.time()

lemmatizer = prepLemmatizer()

if index_and_export:
    py_files, error_files = indexPyFiles(lemmatizer)
    exportFiles(py_files, error_files)

if import_and_search:
    py_files = importFiles()
    searchFiles(py_files, lemmatizer)
    

print("\nFinished in %s seconds" % (time.time() - start_time))