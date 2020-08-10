# import xml parser
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_and_get_ns(file):
    events = "start", "start-ns"
    root = None
    ns = {}
    for event, elem in ET.iterparse(file, events):
        if event == "start-ns":
            ns[elem[0] if len(elem[0]) > 0 else 'sentic'] = elem[1]
        elif event == "start":
            if root is None:
                root = elem
    return ET.ElementTree(root), ns

class SenticNetConcept(object):
    """ A Senticnet Concept holding all relevant information about it """

    def __init__(self, child, i, ns):
        self.index = i
        # extract from child
        self.url = child.attrib["{%s}about" % ns['rdf']]
        self.text = child.find("sentic:text", ns).text.replace(' ', '_')
        # get sentic attributes
        sentics = child.find("sentic:sentics", ns)
        self.pleasentness = float(sentics.find("sentic:pleasantness", ns).text)
        self.sensitivity = float(sentics.find("sentic:sensitivity", ns).text)
        # get semantics
        self.semantic_urls = [
            s.attrib['{%s}resource' % ns['rdf']]        
            for s in child.find("sentic:semantics", ns)
        ]

class SenticNetGraph(object):
    """ SenticNet Graph
        Expecting .rdf.xml file
    """

    def __init__(self, path:str):
        print("Loading SenticNet Graph...", end=" ")
        # load xml file and get root
        tree, self.ns = parse_and_get_ns(path)
        root = tree.getroot()
        # get list of all concepts and create a mapping from url to concept
        self.concepts = [SenticNetConcept(child, i, self.ns) for i, child in enumerate(root)]
        self.url2concept_id = {c.url: c.index for c in self.concepts}
        self.base_url = '/'.join(self.concepts[0].url.split('/')[:-1])
        # delete xml tree
        del tree, root

        print("Done")

    def get_node_from_id(self, i):
        return self.concepts[i]

    def get_concept(self, term:str):
        # build url to concept
        url = "%s/%s" % (self.base_url, term)
        # check if url is valid
        if url not in self.url2concept_id:
            return None
        # get id and return concept
        return self.concepts[self.url2concept_id[url]]

    def get_semantic_ids(self, concept:SenticNetConcept):
        # get semantics from url
        return [self.url2concept_id[url] for url in concept.semantic_urls if url in self.url2concept_id]