import json
from flask import Flask,request,jsonify

app = Flask(__name__)

import tensorflow.compat.v1 as tf
import os 
import shutil
import csv
import pandas as pd
import IPython

print("Libraries Initialised!")

tf.get_logger().setLevel('ERROR')

from tapas-flask-api.tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.scripts import prediction_utils

print("Imported tapas source code!")

os.makedirs('results/sqa/tf_examples', exist_ok=True)
os.makedirs('results/sqa/model', exist_ok=True)
with open('results/sqa/model/checkpoint', 'w') as f:
  f.write('model_checkpoint_path: "model.ckpt-0"')
for suffix in ['.data-00000-of-00001', '.index', '.meta']:
  shutil.copyfile(f'tapas_sqa_base/model.ckpt{suffix}', f'results/sqa/model/model.ckpt-0{suffix}')


print("Results folder ready!")

max_seq_length = 512
vocab_file = "tapas_sqa_base/vocab.txt"
config = tf_example_utils.ClassifierConversionConfig(
    vocab_file=vocab_file,
    max_seq_length=max_seq_length,
    max_column_id=max_seq_length,
    max_row_id=max_seq_length,
    strip_column_names=False,
    add_aggregation_candidates=False,
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)

print("Vocab set up!")

def convert_interactions_to_examples(tables_and_queries):
  """Calls Tapas converter to convert interaction to example."""
  for idx, (table, queries) in enumerate(tables_and_queries):
    interaction = interaction_pb2.Interaction()
    for position, query in enumerate(queries):
      question = interaction.questions.add()
      question.original_text = query
      question.id = f"{idx}-0_{position}"
    for header in table[0]:
      interaction.table.columns.add().text = header
    for line in table[1:]:
      row = interaction.table.rows.add()
      for cell in line:
        row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    for i in range(len(interaction.questions)):
      try:
        yield converter.convert(interaction, i)
      except ValueError as e:
        print(f"Can't convert interaction: {interaction.id} error: {e}")
        
def write_tf_example(filename, examples):
  with tf.io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def predict(table_data, queries):
  print("Prediction started!")	
  table = [list(map(lambda s: s.strip(), row.split("|"))) 
           for row in table_data.split("\n") if row.strip()]
  examples = convert_interactions_to_examples([(table, queries)])
  write_tf_example("results/sqa/tf_examples/test.tfrecord", examples)
  write_tf_example("results/sqa/tf_examples/random-split-1-dev.tfrecord", [])

  print("Processed table data!")

  os.system(''' python tapas/tapas/run_task_main.py \
    --task="SQA" \
    --output_dir="results" \
    --noloop_predict \
    --test_batch_size=3 \
    --tapas_verbosity="ERROR" \
    --compression_type= \
    --init_checkpoint="tapas_sqa_base/model.ckpt" \
    --bert_config_file="tapas_sqa_base/bert_config.json" \
    --mode="predict" 2> error''')

  print("Prediction completed!")

  results_path = "results/sqa/model/test_sequence.tsv"
  all_coordinates = []
  answers_lst=[]
  df = pd.DataFrame(table[1:], columns=table[0])
  #display(IPython.display.HTML(df.to_html(index=False)))
  print("Result printing!")
  with open(results_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
      all_coordinates.append(coordinates)
      answers = ', '.join([table[row + 1][col] for row, col in coordinates])
      position = int(row['position'])
      print(">", queries[position])
      print(answers)
      answers_lst.append(answers)
  return answers_lst



def predict_single_question(table_data, queries):
  print("Prediction started!")	
  table = [list(map(lambda s: s.strip(), row.split("|"))) 
           for row in table_data.split("\n") if row.strip()]
  examples = convert_interactions_to_examples([(table, queries)])
  write_tf_example("results/sqa/tf_examples/test.tfrecord", examples)
  write_tf_example("results/sqa/tf_examples/random-split-1-dev.tfrecord", [])

  print("Processed table data!")

  os.system(''' python tapas/tapas/run_task_main.py \
    --task="SQA" \
    --output_dir="results" \
    --noloop_predict \
    --test_batch_size=1 \
    --tapas_verbosity="ERROR" \
    --compression_type= \
    --init_checkpoint="tapas_sqa_base/model.ckpt" \
    --bert_config_file="tapas_sqa_base/bert_config.json" \
    --mode="predict" 2> error''')

  print("Prediction completed!")

  results_path = "results/sqa/model/test_sequence.tsv"
  all_coordinates = []
  answers_lst=[]
  df = pd.DataFrame(table[1:], columns=table[0])
  #display(IPython.display.HTML(df.to_html(index=False)))
  print("Result printing!")
  with open(results_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
      all_coordinates.append(coordinates)
      answers = ', '.join([table[row + 1][col] for row, col in coordinates])
      position = int(row['position'])
      print(">", queries[position])
      print(answers)
      answers_lst.append(answers)
  return answers_lst

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


@app.route('/')
def hello():    
    return "APP is running!"

@app.route('/tapas-demo')
def ask():
    q=request.args['q']
    result = predict("""
    Pos | No | Driver               | Team                           | Laps | Time/Retired | Grid | Points
    1   | 32 | Patrick Carpentier   | Team Player's                  | 87   | 1:48:11.023  | 1    | 22    
    2   | 1  | Bruno Junqueira      | Newman/Haas Racing             | 87   | +0.8 secs    | 2    | 17    
    3   | 3  | Paul Tracy           | Team Player's                  | 87   | +28.6 secs   | 3    | 14
    4   | 9  | Michel Jourdain, Jr. | Team Rahal                     | 87   | +40.8 secs   | 13   | 12
    5   | 34 | Mario Haberfeld      | Mi-Jack Conquest Racing        | 87   | +42.1 secs   | 6    | 10
    6   | 20 | Oriol Servia         | Patrick Racing                 | 87   | +1:00.2      | 10   | 8 
    7   | 51 | Adrian Fernandez     | Fernandez Racing               | 87   | +1:01.4      | 5    | 6
    8   | 12 | Jimmy Vasser         | American Spirit Team Johansson | 87   | +1:01.8      | 8    | 5
    9   | 7  | Tiago Monteiro       | Fittipaldi-Dingman Racing      | 86   | + 1 Lap      | 15   | 4
    10  | 55 | Mario Dominguez      | Herdez Competition             | 86   | + 1 Lap      | 11   | 3
    11  | 27 | Bryan Herta          | PK Racing                      | 86   | + 1 Lap      | 12   | 2
    12  | 31 | Ryan Hunter-Reay     | American Spirit Team Johansson | 86   | + 1 Lap      | 17   | 1
    13  | 19 | Joel Camathias       | Dale Coyne Racing              | 85   | + 2 Laps     | 18   | 0
    14  | 33 | Alex Tagliani        | Rocketsports Racing            | 85   | + 2 Laps     | 14   | 0
    15  | 4  | Roberto Moreno       | Herdez Competition             | 85   | + 2 Laps     | 9    | 0
    16  | 11 | Geoff Boss           | Dale Coyne Racing              | 83   | Mechanical   | 19   | 0
    17  | 2  | Sebastien Bourdais   | Newman/Haas Racing             | 77   | Mechanical   | 4    | 0
    18  | 15 | Darren Manning       | Walker Racing                  | 12   | Mechanical   | 7    | 0
    19  | 5  | Rodolfo Lavin        | Walker Racing                  | 10   | Mechanical   | 16   | 0
    """, ["what were the team names?",
      "of these, which points did Mario Haberfeld and Oriol Servia score?",
      "who scored 2?"])
   
    print(result)
    output = json.dumps(result, default=set_default)
    return output

@app.route('/tapas-ask')
def tapasAsk():
    q=request.args['q']
    result = predict_single_question("""
    Pos | No | Driver               | Team                           | Laps | Time/Retired | Grid | Points
    1   | 32 | Patrick Carpentier   | Team Player's                  | 87   | 1:48:11.023  | 1    | 22    
    2   | 1  | Bruno Junqueira      | Newman/Haas Racing             | 87   | +0.8 secs    | 2    | 17    
    3   | 3  | Paul Tracy           | Team Player's                  | 87   | +28.6 secs   | 3    | 14
    4   | 9  | Michel Jourdain, Jr. | Team Rahal                     | 87   | +40.8 secs   | 13   | 12
    5   | 34 | Mario Haberfeld      | Mi-Jack Conquest Racing        | 87   | +42.1 secs   | 6    | 10
    6   | 20 | Oriol Servia         | Patrick Racing                 | 87   | +1:00.2      | 10   | 8 
    7   | 51 | Adrian Fernandez     | Fernandez Racing               | 87   | +1:01.4      | 5    | 6
    8   | 12 | Jimmy Vasser         | American Spirit Team Johansson | 87   | +1:01.8      | 8    | 5
    9   | 7  | Tiago Monteiro       | Fittipaldi-Dingman Racing      | 86   | + 1 Lap      | 15   | 4
    10  | 55 | Mario Dominguez      | Herdez Competition             | 86   | + 1 Lap      | 11   | 3
    11  | 27 | Bryan Herta          | PK Racing                      | 86   | + 1 Lap      | 12   | 2
    12  | 31 | Ryan Hunter-Reay     | American Spirit Team Johansson | 86   | + 1 Lap      | 17   | 1
    13  | 19 | Joel Camathias       | Dale Coyne Racing              | 85   | + 2 Laps     | 18   | 0
    14  | 33 | Alex Tagliani        | Rocketsports Racing            | 85   | + 2 Laps     | 14   | 0
    15  | 4  | Roberto Moreno       | Herdez Competition             | 85   | + 2 Laps     | 9    | 0
    16  | 11 | Geoff Boss           | Dale Coyne Racing              | 83   | Mechanical   | 19   | 0
    17  | 2  | Sebastien Bourdais   | Newman/Haas Racing             | 77   | Mechanical   | 4    | 0
    18  | 15 | Darren Manning       | Walker Racing                  | 12   | Mechanical   | 7    | 0
    19  | 5  | Rodolfo Lavin        | Walker Racing                  | 10   | Mechanical   | 16   | 0
    """, [q])
   
    print(q)
    output = json.dumps(result, default=set_default)
    return output

@app.route('/users')
def usersdata():
    q=request.args['q']
    result = predict_single_question("""
    id|userName|name|surname|email|emailConfirmed|phoneNumber|status|createdOn|updatedOn
    57fc1de4-b7cc-4a42-9ddf-39f681e9c66c|Rahul.Ramesh@gds.ey.com|Rahul|R|Rahul.Ramesh@gds.ey.com|FALSE|46453600000000|APPROVED|2020-07-21T04:58:39.6163387|2020-10-09T09:08:33.004867
    edf13035-8493-e8a3-226d-39f6647fa327|Pratima.Behera@gds.ey.com|Pratima|Behera|Pratima.Behera@gds.ey.com|FALSE|9448039770|APPROVED|2020-07-15T11:53:44.5063382|2020-10-09T09:07:56.4323248
    c267b865-e1b5-cc12-4f6b-39f663e026ce|Amina.Thaha1@gds.ey.com|Amina|M Thaha1|Amina.Thaha1@gds.ey.com|FALSE|919996000000|APPROVED|2020-07-15T08:59:32.5754108|2020-10-09T05:33:50.6272115
    0f12620f-859d-651b-2414-39f7f564389d|Vani.Mohan@gds.ey.com|Vani|Mohan|Vani.Mohan@gds.ey.com|FALSE|919886000000|APPROVED|2020-10-01T08:11:31.3644562|2020-10-06T12:55:42.1155515
    c8cc74a5-6f98-e54f-c63d-39f6ca3bb659|Rashmi.Sachin@gds.ey.com|Rashmi|Sachin|Rashmi.Sachin@gds.ey.com|FALSE||APPROVED|2020-08-04T06:00:48.9870037|2020-10-06T06:12:00.4539584
    e4ead65b-a6dc-14ec-0fb3-39f7f585ca54|Karthika.LR@gds.ey.com|Karthika|Lr|Karthika.LR@gds.ey.com|FALSE|918589000000|APPROVED|2020-10-01T08:48:11.385674|2020-10-01T08:48:19.6155881
    c0108f6c-b903-a2eb-ced1-39f7f583c065|Trung.D.Cao@de.ey.com|Trung Duc|Cao|Trung.D.Cao@de.ey.com|FALSE|4916090000000|REVOKED|2020-10-01T08:45:57.7377365|2020-10-01T08:47:40.7448312
    6c10bb22-6921-5589-f61e-39f7f57d8cd4|Anil.S@gds.ey.com|Anil|S|Anil.S@gds.ey.com|FALSE|919568000000|APPROVED|2020-10-01T08:39:11.3223006|2020-10-01T08:39:24.4761895
    c04d1ec5-2bef-1409-514a-39f7f55f66e9|Akshada.Nayak@de.ey.com|Akshada|Nayak|Akshada.Nayak@de.ey.com|FALSE|4916090000000|REVOKED|2020-10-01T08:06:15.539297|2020-10-01T08:12:03.1381272
    f4b4ce02-c22d-ad00-e799-39f7f56231b1|martin.rothenhaeusler@de.ey.com|Martin|Rothenh„usler|martin.rothenhaeusler@de.ey.com|FALSE|4916090000000|REVOKED|2020-10-01T08:09:18.5216095|2020-10-01T08:11:58.2485097
    20aa420c-87b7-49c8-4e7b-39f7f5475684|Jenny.W.Le@de.ey.com|Jenny|Le|Jenny.W.Le@de.ey.com|FALSE|4916090000000|APPROVED|2020-10-01T07:39:58.4782203|2020-10-01T07:40:07.3163481
    4117d8d4-9f76-67ca-90bf-39f66484bd11|Diana.Francis@gds.ey.com|Diana|Francis|Diana.Francis@gds.ey.com|FALSE|75765675756|APPROVED|2020-07-15T11:59:18.8223446|2020-09-24T06:21:21.8345063

    """, [q])
   
    print(q)
    output = json.dumps(result, default=set_default)
    return output



if __name__=='__main__':
    app.run() 




