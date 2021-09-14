from xml.dom import minidom


class DeductorParser:
    def __init__(self, filename=None):
        self.dom = None
        self.docs_xml = None
        self.document_xml = None
        self.neuronets_xml = {}
        self.neuro_xml = {}
        self.layers = []
        self.layers_count = 0
        self.count_x = 0
        self.count_y = 0
        self.krutizna = 0
        self.activate = ''
        self.inp_norm_from = []
        self.inp_norm_to = []
        self.out_norm_from = []
        self.out_norm_to = []
        self.inp_names = []
        self.out_names = []
        self.filename = ''
        self.document_name = ''
        self.neuro_name = ''
        self.neuro_title = ''
        self.neuro_description = ''
        if not (filename is None):
            self.loadfile(filename)

    def loadfile(self, filename):
        self.filename = filename
        dom = minidom.parse(filename)
        dom.normalize()
        self.dom = dom

    def loaddocumentsxml(self):
        document = self.dom.getElementsByTagName('Document')[0]
        script = document.getElementsByTagName('Script')[0]
        SubNodes = script.getElementsByTagName('SubNodes')[0]
        NodeCount = int(SubNodes.getElementsByTagName('Count')[0].childNodes[0].nodeValue)
        docs = {}
        # docs_lst = [SubNodes.getElementsByTagName(f'I_{i}')[0] for i in range(NodeCount)]
        for i in range(NodeCount):
            el_list = SubNodes.getElementsByTagName(f'I_{i}')
            for el in el_list:
                if el.parentNode == SubNodes:
                    name = el.getElementsByTagName('DisplayName')[0].childNodes[0].nodeValue
                    docs[name] = el
                    break
        self.docs_xml = docs

    def getdocumentsxml(self, load=True):
        if load:
            self.loaddocumentsxml()
        return self.docs_xml

    def setdocument(self, name, load=True):
        if load:
            self.loaddocumentsxml()
        if name in self.docs_xml:
            self.document_xml = self.docs_xml[name]
            self.document_name = name

    def getneuronetsxml(self, load=True):
        if load:
            self.loadneuronetssxml()
        return self.neuronets_xml

    def loadneuronetssxml(self):
        d = self.document_xml
        SubNodes = d.getElementsByTagName('SubNodes')[0]
        nets = {}
        nets_n = 0
        for el in SubNodes.childNodes:
            if el.nodeName == f'I_{nets_n}':
                name = el.getElementsByTagName('DisplayName')[0].childNodes[0].nodeValue
                if el.getElementsByTagName('VendorName')[0].childNodes[0].nodeValue == 'TBGNeuralNetTeachEngineVendor':
                    nets[name] = el
                nets_n += 1
        self.neuronets_xml = nets

    def setneuronet(self, name, load=True, parse=False):
        if load:
            self.loadneuronetssxml()
        if name in self.neuronets_xml:
            self.neuro_xml = self.neuronets_xml[name]
            self.neuro_title = name
            if parse:
                self.parseneuro()

    def parseneuro(self):
        comp = self.neuro_xml.getElementsByTagName('AnalyticEngine')[0]
        Kernel = self.neuro_xml.getElementsByTagName('Kernel')[0]
        self.krutizna = float(Kernel.getElementsByTagName('Slope')[0].childNodes[0].nodeValue)
        self.activate = Kernel.getElementsByTagName('FunctionType')[0].childNodes[0].nodeValue
        Layers = self.neuro_xml.getElementsByTagName('Layers')[0]
        self.layers_count = int(Layers.getElementsByTagName('Count')[0].childNodes[0].nodeValue) - 1
        self.layers = [[] for _ in range(self.layers_count)]
        layer_n = 0
        for el in Layers.childNodes:
            if el.nodeName == f'I_{layer_n}':
                Neurons = el.getElementsByTagName('Neurons')[0]
                count = int(Neurons.getElementsByTagName('Count')[0].childNodes[0].nodeValue)
                if layer_n:
                    n = 0
                    for neuron in Neurons.childNodes:
                        if neuron.nodeName == f'I_{n}':
                            shift = float(neuron.getElementsByTagName('Shift')[0].childNodes[0].nodeValue)
                            w = [shift]
                            Links = neuron.getElementsByTagName('Links')[0]
                            count_w = int(Links.getElementsByTagName('Count')[0].childNodes[0].nodeValue)
                            w_n = 0
                            for Weight in Links.childNodes:
                                if Weight.nodeName == f'I_{w_n}':
                                    val = float(Weight.getElementsByTagName('Weight')[0].childNodes[0].nodeValue)
                                    w.append(val)
                                    w_n += 1
                            self.layers[layer_n - 1].append(w)
                            n += 1
                            if n == count:
                                break
                else:
                    self.count_x = count
                layer_n += 1
                if layer_n > self.layers_count:
                    break
        self.count_y = len(self.layers[-1])
        Normalizers = self.neuro_xml.getElementsByTagName('Normalizers')[0]
        norm_count = int(Normalizers.getElementsByTagName('Count')[0].childNodes[0].nodeValue)
        norm_n = 0
        norms = []
        for el in Normalizers.childNodes:
            if el.nodeName == f'I_{norm_n}':
                Normalizer = el.getElementsByTagName('Normalizer')[0]
                in_mn, in_mx = None, None
                ExpandMin = Normalizer.getElementsByTagName('ExpandMin')
                ExpandMax = Normalizer.getElementsByTagName('ExpandMax')
                if ExpandMin:
                    in_mn = float(ExpandMin[0].childNodes[0].nodeValue)
                if ExpandMax:
                    in_mx = float(ExpandMax[0].childNodes[0].nodeValue)
                LinearNormalizer = Normalizer.getElementsByTagName('LinearNormalizer')
                if LinearNormalizer:
                    LinearNormalizer = LinearNormalizer[0]
                    enabled = LinearNormalizer.getElementsByTagName('Enabled')
                    if enabled:
                        enabled = int(enabled[0].childNodes[0].nodeValue)
                        if enabled:
                            mn = LinearNormalizer.getElementsByTagName('Min')
                            if mn:
                                mn = float(mn[0].childNodes[0].nodeValue)
                            else:
                                mn = 0
                            mx = LinearNormalizer.getElementsByTagName('Max')
                            if mx:
                                mx = float(mx[0].childNodes[0].nodeValue)
                            else:
                                mx = 0
                            norms.append([[in_mn, in_mx], [mn, mx]])
                        else:
                            norms.append(None)
                    else:
                        norms.append(None)
                else:
                    norms.append(None)
                norm_n += 1
                if norm_n == norm_count:
                    break
        Statistics = self.neuro_xml.getElementsByTagName('Statistics')[0]
        Statistics_items = Statistics.getElementsByTagName('Items')[0]
        Statistics_count = int(Normalizers.getElementsByTagName('Count')[0].childNodes[0].nodeValue)
        stat_n = 0
        stats = []
        for el in Statistics_items.childNodes:
            if el.nodeName == f'I_{stat_n}':
                mn = el.getElementsByTagName('Min')
                if mn:
                    mn = float(mn[0].childNodes[0].nodeValue)
                else:
                    mn = None
                mx = el.getElementsByTagName('Max')
                if mx:
                    mx = float(mx[0].childNodes[0].nodeValue)
                else:
                    mx = None
                stats.append([mn, mx])
                stat_n += 1
                if stat_n == Statistics_count:
                    break
        InputColumns = Kernel.getElementsByTagName('InputColumns')[0]
        self.inp_norm_from = []
        self.inp_norm_to = []
        self.inp_names = []
        inp_n = 0
        for el in InputColumns.childNodes:
            if el.nodeName == f'I_{inp_n}':
                name = el.getElementsByTagName('Name')[0].childNodes[0].nodeValue
                self.inp_names.append(name)
                NormalizerIndex = int(el.getElementsByTagName('NormalizerIndex')[0].childNodes[0].nodeValue)
                val = stats[NormalizerIndex - 1]
                if val[0] is None:
                    val[0] = 0
                if val[1] is None:
                    val[1] = 1
                val_from_norm = norms[NormalizerIndex - 1][0]
                val_to_norm = norms[NormalizerIndex - 1][1]
                if val_from_norm[0] is None:
                    val_from_norm[0] = val[0]
                if val_from_norm[1] is None:
                    val_from_norm[1] = val[1]
                if val_to_norm[0] is None:
                    val_to_norm[0] = -1
                if val_to_norm[1] is None:
                    val_to_norm[1] = 1
                self.inp_norm_from.append(val_from_norm)
                self.inp_norm_to.append(val_to_norm)
                inp_n += 1
                if inp_n == self.count_x:
                    break
        OutputColumns = Kernel.getElementsByTagName('OutputColumns')[0]
        self.out_norm_from = []
        self.out_norm_to = []
        self.out_names = []
        out_n = 0
        for el in OutputColumns.childNodes:
            if el.nodeName == f'I_{out_n}':
                name = el.getElementsByTagName('Name')[0].childNodes[0].nodeValue
                self.out_names.append(name)
                NormalizerIndex = int(el.getElementsByTagName('NormalizerIndex')[0].childNodes[0].nodeValue)
                val = stats[NormalizerIndex - 1]
                if val[0] is None:
                    val[0] = 0
                if val[1] is None:
                    val[1] = 1
                val_to_norm = norms[NormalizerIndex - 1][0]
                val_from_norm = norms[NormalizerIndex - 1][1]
                if val_to_norm[0] is None:
                    val_to_norm[0] = val[0]
                if val_to_norm[1] is None:
                    val_to_norm[1] = val[1]
                if val_from_norm[0] is None:
                    val_from_norm[0] = 0
                if val_from_norm[1] is None:
                    val_from_norm[1] = 1
                self.out_norm_from.append(val_from_norm)
                self.out_norm_to.append(val_to_norm)
                out_n += 1
                if out_n == self.count_y:
                    break
        try:
            self.neuro_description =  self.neuro_xml.getElementsByTagName('Description')[0].childNodes[0].nodeValue
            self.neuro_name = self.neuro_xml.getElementsByTagName('Name')[0].childNodes[0].nodeValue
        except BaseException:
            pass

    @property
    def neurodata(self):
        return {
            'layers': self.layers,
            'layers_count': self.layers_count,
            'count_x': self.count_x,
            'count_y': self.count_y,
            'krutizna': self.krutizna,
            'activate': self.activate,
            'inp_norm_from': self.inp_norm_from,
            'inp_norm_to': self.inp_norm_to,
            'out_norm_from': self.out_norm_from,
            'out_norm_to': self.out_norm_to,
            'inp_names': self.inp_names,
            'out_names': self.out_names,
            'file': self.filename,
            'document': self.document_name,
            'neuronet_name': self.neuro_name,
            'neuronet_description': self.neuro_description,
            'neuronet_title': self.neuro_title,
        }

    @property
    def documents(self, load=True):
        return list(self.getdocumentsxml(load))

    @property
    def neuronets(self, load=True):
        return list(self.getneuronetsxml(load))
