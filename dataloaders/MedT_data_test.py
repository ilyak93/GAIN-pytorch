from dataloaders.MedTData import MedT_Loader

medt_loader = MedT_Loader('C:/MDT_dataset/SB3_ulcers_mined_roi_mult/')

''' infinite for
for sample in medt_loader.datasets['train']:
    s = sample
    if s[1].size() == s[0].size()[0:3]:
        print('with_mask')
    else :
        print('without_mask')
'''
for sample in medt_loader.datasets['test']:
    s = sample
    if s[1].size() == s[0].size()[0:3]:
        print('with_mask')
    else :
        print('without_mask')
        if s[2] == 1:
            print('Pos')
        else:
            print('Neg')

print()