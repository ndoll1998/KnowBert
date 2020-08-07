# import xml parser
import xml.etree.ElementTree as ET
from tqdm import tqdm

class SenticNetConcept(object):
    """ A Senticnet Concept holding all relevant information about it """

    def __init__(self, child, i):
        self.index = i
        # extract from child
        self.url = child.attrib["{%s}about" % SenticNetGraph.ns['rdf']]
        self.text = child.find("sentic:text", SenticNetGraph.ns).text.replace(' ', '_')
        # get semantics
        self.semantic_urls = [
            s.attrib['{%s}resource' % SenticNetGraph.ns['rdf']]        
            for s in child.find("sentic:semantics", SenticNetGraph.ns)
        ]

class SenticNetGraph(object):
    """ SenticNet Graph
        Expecting .rdf.xml file
    """

    # xml namespaces
    ns = {
        'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        'sentic': "http://sentic.net"
    }

    def __init__(self, path:str):
        # load xml file and get root
        tree = ET.parse(path)
        root = tree.getroot()
        # get list of all concepts and create a mapping from url to concept
        self.concepts = [SenticNetConcept(child, i) for i, child in enumerate(root)]
        self.url2concept_id = {c.url: c.index for c in self.concepts}
        # delete xml tree
        del tree, root

    def get_node_from_id(self, i):
        return self.concepts[i]

    def get_concept(self, term:str):
        # build url to concept
        url = "http://sentic.net/api/en/concept/%s" % term
        # check if url is valid
        if url not in self.url2concept_id:
            return None
        # get id and return concept
        return self.concepts[self.url2concept_id[url]]

    def get_semantic_ids(self, concept:SenticNetConcept):
        # get semantics from url
        return [self.url2concept_id[url] for url in concept.semantic_urls]