'''
This is a script which parses an iTell intelligent textbook markdown directory.
It outputs a .csv file containing all subsections of the intelligent textbook.
It takes two inputs: 
The first is a path to the contents page of an iTell intelligent textbook markdown directory.
The second is a path to the output .csv file

You can run this script with a command like
python parse-textbook.py /home/jovyan/active-projects/macro-economics-textbook/contents/ ./subsections.csv
'''

import re
import pandas as pd
from pathlib import Path
import string
import markdown
from bs4 import BeautifulSoup
import string
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('contents_path', type=str, default='/home/jovyan/active-projects/macro-economics-textbook/contents/', help='The path to the top level of the contents directory tree')
    parser.add_argument('output_path', type=str, default='./subsections.csv', help='The output .csv file')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    return parser.parse_args()

verbose = False
subsections_to_skip = []
#[
#     'learn with videos',
#     'please write your summary below',
#     'please your write summary below', # ... a perfect example of why this approach is doomed. More than 10 sections have this typo.
#     'bring it home',
#     'clear it up',
#     'work it out',
# ]


# Define functions

def diff(text, clean_text):
    text_lines = [s.strip() for s in text.splitlines() if s.strip()] # delete empty lines
    clean_text_lines = [s.strip() for s in clean_text.splitlines() if s.strip()] # delete empty lines
    diff = differ.make_file(text_lines, clean_text_lines, fromdesc='Original', todesc='HTML Parsed', context=False, numlines=0)

def clean_md(text_md):
    text_md = exclamation_links_pattern.sub(r'[', text_md)
    text_md = html_table_pattern.sub('', text_md)
    text_md = javascript_import_pattern.sub('', text_md)
    return text_md.strip()

def clean_raw_text(subsection_text_mdx):
    subsection_text_md = clean_md(subsection_text_mdx)
    subsection_text_html = markdown.markdown(subsection_text_md, extensions=['extra', 'tables'])
    subsection_text_html = html_table_pattern.sub('', subsection_text_html) # this gets any markdown tables now been converted to HTML
    return BeautifulSoup(subsection_text_html, features='html.parser').get_text().strip()

def generate_slugs(heading_list):
    remove_these = string.punctuation + '—”“'
    slug_list1 = [heading.lower().replace('-', ' ').translate(str.maketrans('', '', remove_these)).replace(' ', '-') for heading in heading_list] 
    slug_list = []
    i = 0
    for word in slug_list1:
        if slug_list1[:i].count(word) < 1:
            slug_list.append(word)
        else:
            slug_list.append(word+'-'+str(slug_list1[:i].count(word)))       
        i += 1
    return slug_list

def make_slug_list(df):
    slug_list = []
    for chapter in df['chapter'].drop_duplicates():
        df1 = df[df['chapter']==chapter]
        for section in df1['section'].drop_duplicates():
            df2 = df1[df1['section']==section]
            slug_list.append(generate_slugs(list(df2['heading'])))
    return sum(slug_list,[])    

if __name__ == '__main__':    
    # Get markdown and infer metadata
    args = get_args()
    PATH_TO_CONTENTS = args.contents_path
    OUTPUT_FILE = args.output_path
    
    CONTENTS = Path(PATH_TO_CONTENTS)

    mdx_sections = []

    for path in sorted(CONTENTS.glob('**/index.mdx')):
        rel_path = path.relative_to('/home/jovyan/active-projects/macro-economics-textbook/contents/')
        parents = reversed(rel_path.parents[:-1]) # get all parent directories before '../contents'. We omit ([:-1]) the top-level directory 
        module = next(parents).name.split('-')[1] # all .mdx files belong to a module
        chapter = next(parents, Path('-0')).name.split('-')[1] # if the iterator is exhausted, this will make the chapter number 0
        section = next(parents, Path('-0')).name.split('-')[1] # if the iterator is exhausted, this will make the section number 0
        if verbose:
            print(f'{path.as_posix():<60}Module {module}, Chapter {chapter}, Section {section}')
        mdx_sections.append({
            'module': module,
            'chapter': chapter,
            'section': section,
            'path': path,
        })

    
    # Parse with Regex    
    ## all these are lowercased because capitalization is inconsistent across MDX files
    pattern = re.compile(r'(?:^#{1,2} )(.*?)$\s*(.*?)(?=\s*^#|\Z)', re.DOTALL | re.MULTILINE)

    subsections = []

    for section in mdx_sections:
        text = section['path'].read_text()
        matches = pattern.findall(text)
        for i, match in enumerate(matches):
            subsection_title = match[0]
            subsection_text = match[1]
            if subsection_title.lower().strip() in subsections_to_skip: # lowercase() and strip() because formating is inconsistent
                if verbose and len(subsection_text) > 10: # if verbose, print the longer sections that we will be EXcluding
                    print('-'*80)
                    print(subsection_title, '\n', subsection_text)
                    print('_'*80)                
                continue
            elif verbose and len(subsection_text) < 100: # if verbose, print the shorter sections that we will still be INcluding
                print('-'*80)
                print(subsection_title, '\n', subsection_text)
                print('_'*80)
            else:
                subsection_dict = {
                    **section, # add in section-level metadata
                    'subsection': i,
                    'heading': subsection_title,
                    'raw_text': subsection_text,
                }
                subsection_dict.pop('path')
                subsections.append(subsection_dict)

    df = pd.DataFrame(subsections)

    # HTML cleanup
    exclamation_links_pattern = re.compile(r'!\[') # Removes the exclamation point in links ![link_text](url) --> [link_text](url)
    html_table_pattern = re.compile(r'<Table.*?</Table>\s*', re.DOTALL | re.IGNORECASE) # remove table HTML and its contents
    javascript_import_pattern = re.compile(r'^import.*?;', re.MULTILINE) # remove javascript imports
    df['clean_text'] = df.raw_text.apply(clean_raw_text)

    # Generate slugs
    df['slug'] = make_slug_list(df)

    # Add unique id for each subsection
    df['id'] = df.apply(lambda row: str(row['chapter']) + '-' + 
                        str(row['section']) + '-' + str(row['subsection']) + 
                        '-' + row['slug'], axis=1)

    df = df[['id', 'module', 'chapter', 'section', 'subsection', 'heading', 'raw_text', 'clean_text', 'slug']]
    df[['module', 'chapter', 'section', 'subsection']] = df[['module', 'chapter', 'section', 'subsection']].astype(int)

    df.to_csv(OUTPUT_FILE, index=False)
