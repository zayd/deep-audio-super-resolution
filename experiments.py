from collections import OrderedDict

experiments = []
#experiments += [OrderedDict([('dataset', 'multispeaker'), ('upsample', 4),
#                ('model', 'dnn'), ('phase', 'cheated'), ('subsample', 0.25)])]
experiments += [OrderedDict([('dataset', 'multispeaker'), ('upsample', 4),
                ('model', 'dnn'), ('phase', 'regression'), ('subsample', 0.25)])]
#experiments = #[OrderedDict([('dataset', 'speaker1'), ('upsample', 2),
               # ('model', 'dnn'), ('phase', 'regression')])]#,
               #OrderedDict([('dataset', 'speaker1'), ('upsample', 4),
               # ('model', 'dnn'), ('phase', 'regression')]),
               #OrderedDict([('dataset', 'speaker1'), ('upsample', 6),
               # ('model', 'dnn'), ('phase', 'regression')]),
               #OrderedDict([('dataset', 'speaker1'), ('upsample', 8),
               # ('model', 'dnn'), ('phase', 'regression')])]
#
#experiments += [OrderedDict([('dataset', 'multispeaker'), ('upsample', 2),
#                ('model', 'dnn'), ('phase', 'imaged'), ('subsample', 0.25)]),
#               OrderedDict([('dataset', 'multispeaker'), ('upsample', 4),
#                ('model', 'dnn'), ('phase', 'imaged'), ('subsample', 0.25)]),
#               OrderedDict([('dataset', 'multispeaker'), ('upsample', 6),
#                ('model', 'dnn'), ('phase', 'imaged'), ('subsample', 0.25)])]

#experiments += [OrderedDict([('dataset', 'music'), ('upsample', 2),
#                ('model', 'dnn'), ('phase', 'imaged')]),
#               OrderedDict([('dataset', 'music'), ('upsample', 4),
#                ('model', 'dnn'), ('phase', 'imaged')]),
#               OrderedDict([('dataset', 'music'), ('upsample', 6),
#                ('model', 'dnn'), ('phase', 'imaged')])]
