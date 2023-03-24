from datasets import load_dataset
from lm_dataformat import *
import argparse
import jsonlines
import tqdm

# Used to process the following datasets:
# python process_to_lm_data_format.py --input_type text --input raw_data/open_subtitles.txt --output_dir opensubtitles.jsonl.zst --name opensubtitles --doc_size 20 --overlap 3
# python process_to_lm_data_format.py --input_type text --input raw_data/ted2020.txt --output_dir ted2020.jsonl.zst --name ted2020 --doc_sep (Applause)
# python process_to_lm_data_format.py --input_type text --input raw_data/bible.txt --output_dir bible.jsonl.zst --name bible-uedin
# python process_to_lm_data_format.py --input_type text --input raw_data/tanzil.txt --output_dir tanzil.jsonl.zst --name tanzil
# python process_to_lm_data_format.py --input_type text --input raw_data/globalvoices.txt --output_dir globalvoices.jsonl.zst --name globalvoices
# python process_to_lm_data_format.py --input_type text --input raw_data/gnome.txt --output_dir gnome.jsonl.zst --name gnome
# python process_to_lm_data_format.py --input_type json --input raw_data/train.json --output_dir soda.jsonl.zst --name soda
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_type", type=str, required=True, choices=["text", "hf", 'json'])
    argparser.add_argument('--input', type=str, required=True, help='Huggingface dataset name or input text file')
    argparser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    argparser.add_argument('--name', type=str, required=True, help='Name of the dataset')
    argparser.add_argument('--doc_size', type=int, default=-1, help='How many lines to put in each document/record')
    argparser.add_argument('--overlap', type=int, default=0, help='How many lines to overlap between documents/records')
    # add optional argument for a document separator token
    argparser.add_argument('--doc_sep', type=str, default='', help='Document separator token. Enabling this option will split the input text into documents based on the separator token and it will override the doc_size argument.')
    
    args = argparser.parse_args()
    if args.doc_sep != '':
        print(f"Separating documents by {args.doc_sep}")

    if args.input_type == "text":
        ar = Archive(args.output_dir)
        lines = []
        start_line = 0
        with open(args.input, "r", encoding="utf-8") as f:
            # add progress bar
            for i,l in enumerate(f):
                lines.append(l.strip())
                if args.doc_size != -1 and args.doc_sep != '':
                    if i % args.doc_size == 0:  
                        ar.add_data('\n'.join(lines), meta={"source": args.name, "starting_line": i-int(args.overlap) if i > 0 else 0, "ending_line": i + (max(0, args.doc_size))})
                        if args.overlap != 0:
                            lines = lines[-int(args.overlap):]
                        else:
                            lines = []
                        
                elif args.doc_sep != '':
                    if l.strip() == args.doc_sep:
                        ar.add_data('\n'.join(lines), meta={"source": args.name, "starting_line": start_line-int(args.overlap) if start_line > 0 else 0, "ending_line": i})
                        if args.overlap != 0:
                            lines = lines[-int(args.overlap):]
                        else:
                            lines = []
                        start_line = i + 1

            if args.doc_size == -1:
                ar.add_data('\n'.join(lines), meta={"source": args.name, "starting_line": 0, "ending_line": i})        
        ar.commit()
    elif args.input_type == "hf":

        dataset = load_dataset(args.input)
        dataset = dataset['train']

        # writer = Writer(args.output_dir)
        for i, row in enumerate(dataset):
            # writer.write_document(row['text'])
            print(row)
            if i % 1000 == 0:
                print(f'Processed {i} documents')

    elif args.input_type == "json":
        # this is for the SODA dataset only
        ar = Archive(args.output_dir)
        with open (args.input, 'r') as f:
            objs = json.loads(f.read())
            for i, key in enumerate(objs['dialogue']):
                speakers = objs['speakers'][key]
                # for speaker in speakers:
                #     if speaker in objs['PersonX'][key] and objs['PersonX'][key] != '':
                #         speakers[speakers.index(speaker)] = 'User1'
                #     elif speaker in objs['PersonY'][key] and objs['PersonY'][key] != '':
                #         speakers[speakers.index(speaker)] = 'User2'
                #     elif speaker in objs['PersonZ'][key] and objs['PersonZ'][key] != '':
                #         speakers[speakers.index(speaker)] = 'User3'
                # remove "" from dialogue utterances objs['dialogue'][key][i]
                # loop through dialogue utterances and remove \" if it is the first and last character of an utterance
                # CHECK: Zuri rode through Denzel's neighborhood to show him that he could ride his bike. He weaved in and out of the houses, narrowly avoiding parked cars and small children playing. When he reached Denzel's house, he triumphantly skidded to a stop, grinning from ear to ear.
                for j, utterance in enumerate(objs['dialogue'][key]):          
                    if len(utterance) > 1 and utterance[0] == '\"' and utterance[-1] == '\"':
                        objs['dialogue'][key][j] = utterance[1:-1]
                formatted_dialogue = '\n'.join([f"{speakers[i]}: {objs['dialogue'][key][i]}" for i in range(len(objs['dialogue'][key]))])
                narrative = objs['narrative'][key]
                # if objs['PersonX'][key] != '':
                #     narrative = narrative.replace(objs['PersonX'][key], 'User1')
                # if objs['PersonY'][key] != '':
                #     narrative = narrative.replace(objs['PersonY'][key], 'User2')
                # if objs['PersonZ'][key] != '':
                #     narrative = narrative.replace(objs['PersonZ'][key], 'User3')
                text = f"{narrative}\n{formatted_dialogue}"
                ar.add_data(text, meta={"source": args.name, 
                    'narrative': objs['narrative'][key],
                    'dialogue': objs['dialogue'][key],
                    'speakers': objs['speakers'][key],
                    'head': objs['head'][key],
                    'relation': objs['relation'][key],
                    'tail': objs['tail'][key],
                    'literal': objs['literal'][key],
                    'PersonX': objs['PersonX'][key],
                    'PersonY': objs['PersonY'][key],
                    'PersonZ': objs['PersonZ'][key],
                    'original_index': objs['original_index'][key],
                    'split': objs['split'][key],
                    'head_answer': objs['head_answer'][key],
                    'pmi_head_answer': objs['pmi_head_answer'][key],
                    'relation_tail_answer': objs['relation_tail_answer'][key],
                    'pmi_relation_tail_answer': objs['pmi_relation_tail_answer'][key],
                    'id': key, 
                })
                if i % 1000 == 0:
                    print(f'Processed {i} documents')
        ar.commit()

if __name__ == '__main__':
    main()
    # dataset = load_dataset('bible-nlp/biblenlp-corpus', languages = ['eng'])
    # # iterate over the dataset
    # print(dataset.info())

