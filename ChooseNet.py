from net.ACRNN import ACRNN
# from net.ACRNN_CEED import ACRNN_CEED
# from net.ACRNN_DREAMER import ACRNN_DREAMER
from net.LGGNet import LGGNet
from net.Conformer import Conformer
from net.Deformer import Deformer
from net.TSception import TSception
from net.DGCNN import DGCNN
from net.DSE_TAM import DSE_TAM
def choose_net(params):
    graph_gen = []
    original_order = []
    if params['dataset'] == 'DEAP':
        original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                          'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                          'P4', 'P8', 'PO4', 'O2']
        graph_gen = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'], ['FC5', 'FC1', 'FC6', 'FC2'],
                     ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'], ['P7', 'P3', 'Pz', 'P4', 'P8'],
                     ['PO3', 'PO4'],
                     ['O1', 'Oz', 'O2'], ['T7'], ['T8']]
    if params['dataset'] == 'DREAMER':
        original_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        graph_gen = [['AF3', 'AF4'], ['F7', 'F3', 'F4', 'F8'], ['FC5', 'FC6'], ['T7'], ['P7', 'P8'], ['O1', 'O2'], ['T8']]
    if params['dataset'] == 'CEED':
        original_order = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3',
                          'P7',
                          'O1', 'Oz',
                          'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
                          'AF7', 'AF3',
                          'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
                          'PO4',
                          'PO8', 'P6',
                          'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz']
        graph_gen = [['Fp1', 'Fp2'], ['AF7', 'AF3', 'AFz', 'AF4', 'AF8'],
                     ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'],
                     ['FT9', 'FT7', 'T7', 'TP7', 'TP9'], ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
                     ['FT8', 'FT10', 'T8', 'TP8', 'TP10'], ['C5', 'C3', 'C1', 'C2', 'C4', 'C6'],
                     ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'], ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],
                     ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'], ['O1', 'Oz', 'O2']]

    graph_idx = graph_gen

    idx = []
    num_chan_local_graph = []
    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(original_order.index(chan))
    net = None
    if params['net'] == "ACRNN":
        net = ACRNN(params['num_electrodes'])
        # net = ACRNN_CEED(params['num_electrodes'])
        # net = ACRNN_DREAMER(params['num_electrodes'])

    if params['net'] == "LGGNet":
        net = LGGNet(params['num_classes'], (1, params['num_electrodes'], 384), 128, 64, 32, 0.5, 16, 0.25, idx, num_chan_local_graph)
    if params['net'] == "Conformer":
        net = Conformer(n_classes=params['num_classes'], n_chan=params['num_electrodes'], n_hidden=800)
    if params['net'] == "Deformer":
        net = Deformer(num_chan=params['num_electrodes'], num_time=384, temporal_kernel=11, num_kernel=64,
                       num_classes=params['num_classes'], depth=4, heads=16,
                       mlp_dim=16, dim_head=16, dropout=0.5)
    if params['net'] == "TSception":
        net = TSception(
            num_classes=params['num_classes'], input_size=(1, params['num_electrodes'], 384),
            sampling_rate=128, num_T=15, num_S=15,
            hidden=32, dropout_rate=params['dropout'])
    if params['net'] == 'DGCNN':
        net = DGCNN(5,
                    num_electrodes=params['num_electrodes'],
                    num_layers=4,
                    hid_channels=64,
                    num_classes=params['num_classes'])
    if params['net'] == 'DSE-TAM':
        net = DSE_TAM(params)
    return net