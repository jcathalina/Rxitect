from typing import Optional, List
import re
from functools import reduce


def atomwise(mol_str: str, exclusive_tokens: Optional[List[str]] = None):
    """
    Tokenize a molecule string (at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
        (3) All other symbols are tokenized on character level.
    Args:
        mol_str (str): A mol_str string
        exclusive_tokens (Optional[List[str]]): A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
        Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """

    regex = '(\[[^\[\]]{1,10}\])'
    char_list = re.split(regex, mol_str)
    tokens = []
        
    if exclusive_tokens:
        for char in char_list:
            if char.startswith('['):
                if char in exclusive_tokens:
                    tokens.append(str(char))
                else:
                    tokens.append('[UNK]')
            else:
                chars = [unit for unit in char]
                [tokens.append(i) for i in chars]                    
        
    if not exclusive_tokens:
        for char in char_list:
            if char.startswith('['):
                tokens.append(str(char))
            else:
                chars = [unit for unit in char]
                [tokens.append(i) for i in chars]
                
    #fix the 'Br' be splited into 'B' and 'r'
    if 'r' in tokens:
        for index, tok in enumerate(tokens):
            if tok == 'r':
                if tokens[index-1] == 'B':
                        tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
        
    #fix the 'Cl' be splited into 'C' and 'l'
    if 'l' in tokens:
        for index, tok in enumerate(tokens):
            if tok == 'l':
                if tokens[index-1] == 'C':
                        tokens[index-1: index+1] = [reduce(lambda i, j: i + j, tokens[index-1 : index+1])]
    return tokens
