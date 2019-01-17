# COBRAS: A tool for analyzing the collaborative writing process using semantic analysis

COBRAS (COllaBorative pRocess Analysis with Semantics) provides analytics algorithms and visualizations for collaborative writing.
It currently handles three collaborative document editors, [Etherpad](http://etherpad.org/), [Collab-react-components](https://github.com/chili-epfl/collab-react-components) and [FROG](https://github.com/chili-epfl/FROG)

## Setup

### Python Environment
You will need Python3 and the following dependencies.
- csv
- ast
- numpy
- argparse
- pymongo (if using collab-react-components editor or FROG)
- matplotlib
- pandas
- seaborn
- sqlite3
- flask (if using the webserver)
- MySQLdb (if connecting to MySQL database)
- spacy (for semantic analysis)
- scipy (for semantic analysis)
- nltk (for semantic analysis)
```
pip install csv
pip install ast
pip install numpy
pip install argparse
pip install pymongo
pip install matplotlib
pip install pandas
pip install seaborn
pip install sqlite3
pip install flask
pip install mysqlclient
```

### Etherpad
You'll need to download etherpad from [http://etherpad.org/](http://etherpad.org/).
Extract it in a folder named etherpad.

You can start etherpad by runnning `start.bat`.
Then connect to [http://localhost:9001](http://localhost:9001).
> Note:  If you want to join the pad on another terminal, replace localhost by the IP address of the host. Users collaborating need to be on the same network or remotely connect to the host.

Start Editing.

All edits are stored in `etherpad/var/dirty.db` by default but it is possible to modify this so that, for example, a MySQL database is used instead. See section MySQL for more information on this.

(changeset format: [http://policypad.readthedocs.io/en/latest/changesets.html](http://policypad.readthedocs.io/en/latest/changesets.html))

> Note: If it is not possible to start etherpad by running `start.bat` it may be possible to start it by running `bin/run.sh`.

### Etherpad-lite with MySQL
After setting up MySQL, it is possible to store the Etherpad edits in a MySQL database instead of in the default `dirty.db` by modifying `settings.json`. More information in: [https://github.com/ether/etherpad-lite/wiki/How-to-use-Etherpad-Lite-with-MySQL](https://github.com/ether/etherpad-lite/wiki/How-to-use-Etherpad-Lite-with-MySQL)

Default:

```
{
  "dbType" : "dirty",
  "dbSettings" : {
    "filename" : "var/dirty.db"
  },
}
```

MySQL example configuration:

```
{
  "dbType" : "mysql",
  "dbSettings" : {
    "user"    : "etherpaduser",
     "host"    : "localhost",
     "port"    : 3306,
     "password": "p4s$w0Rd",
     "database": "etherpad_lite_db",
     "charset" : "utf8mb4"
  },
}
```

### Collab-react-components
In order to use the text collab editor written by [Dario Anongba](https://github.com/darioAnongba). First install pymango by using:
```
pip install pymongo
```
Clone the program from Dario:  [Collab-React-Components](https://github.com/chili-epfl/collab-react-components) in any folder (doesn't have to be in the same folder as the repo).
Install node.js/npm
Install react with npm (version 15.4.0+) in Dario's folder
```
git clone https://github.com/chili-epfl/collab-react-components
npm install --save react
npm install --save react-dom
```

`npm init` and `npm install` in root folder and in the demos folder
```
npm init
npm install
cd demos/collab-editor
npmn init
npm install
```
Replace `server.js` with the corresponding file in this repo so that the edits are stored in the mongo database. Then start the program with `npm start`
```
service mongod start
cd demos/collab-editor
npm start
```

To access the edits, the program will use pymongo.
To delete the logs, you need to run the python script `drop mongo database.py`
```
python "drop mongo dabase.py"
```


Find a detailed tutorial on [MongoDB](http://api.mongodb.com/python/current/tutorial.html)
(op format [https://github.com/ottypes/text](https://github.com/ottypes/text))

## Usage

In its current state, the program groups the fine grained writing events (`ElementaryOperation`) provided by the editors into `Operations` (list of writing events by the same author in a short time lapse and at the same position). Then `Paragraphs` (collection of operations in the same line) and `SuperParagraphs` (collection of `Paragraphs` that are separated by at least two newline characters) are deduced. We can then fill in information about the `Operation`, such as the `Paragraph` index where it takes place.

The `Operation` information is the following:

| Information          | Details                                      |
|----------------------|----------------------------------------------|
| Author               | Name of the Operation author.                |
| Position Start       | First text position edited.                  |
| Position End         | Last text position edited.                   |
| Time Start           | Operation starting timestamp.                |
| Time End             | Operation ending timestamp.                  |
| Atomic Op Count      | Number of atomic operations.                 |
| Type                 | Operation type.                              |
| Text Added           | Number of characters added.                  |
| Deletion Length      | Number of characters deleted.                |
| Paragraph            | Paragraph current index.                     |
| Paragraph History    | Paragraph persistent identifier.             |
| Paragraph Original   | Simplified paragraph persistent identifier.  |
| Superparagraph ID    | Computed as Paragraph persistent identifier. |
| Coauthor Number      | Number of other authors in SuperParagraph.   |
| Proportion Pad       | Ratio of Operation chars to Total Pad chars. |
| Proportion Paragraph | Ratio of Operation chars to Paragraph chars. |


After completing the information about the `Operations` we can then compute several metrics from the context of an operation:
- Proportion score: how balanced are the contributions of the users in the whole document? (1 is very balanced, 0 unbalanced)
- User proportion per paragraph score: On average over all the paragraphs, how balanced are the contributions of the users per paragraph? (1 is very balanced, 0 unbalanced)
- Synchronous score: how synchronous are the contributions of the users in the whole document? (1 is synchronous, 0 unsynchronous)
- Alternating score: Are the main author of each paragraph alternating? (1 if the main author of each paragraph is always different than the main author of the next paragraph. Very close to 0 if each author wrote a block of paragraph in a sequence)
- Break score day: Do they often take long breaks? If no operation as been made within 8 hours in the pad, then we increment the number of day breaks. This score tells whether users wrote the pad in one day or several ones. The scores tends to increase as the number of “day break” increases.
- Break score short: Do they often take short breaks? Same as above, except we consider a short break to be 10 minutes.
- Overall write type score: What is the proportion of 'write' operations? (writing a relatively big amount of letters)
- Overall paste type score: What is the proportion of 'paste' operations? (from copy/paste)
- Overall delete type score: What is the proportion of 'delete' operations? (deleting a relatively big amount of letters)
- Overall edit type score: What is the proportion of 'edit' operations? (deleting, adding or changing a relatively small amount of letters)
- User write score: How balanced are the 'write' operations? (1 if every user has the same proportion of 'write' operations. 0 if a user did all the 'write' operations)
- User paste score: How balanced are the 'paste' operations? (1 if every user has the same proportion of 'paste' operations. 0 if a user did all the 'paste' operations)
- User delete score: How balanced are the 'delete' operations? (1 if every user has the same proportion of 'delete' operations. 0 if a user did all the 'delete' operations)
- User edit score: How balanced are the 'edit' operations? (1 if every user has the same proportion of 'edit' operations. 0 if a user did all the 'edit' operations)
- Length all write: Total characters written (not pasted).
- Length all paste: Total characters pasted.
- Added characters: All characters added in total.
- Deleted characters: All characters deleted in total.
- Paragraph average length: Average length in characters considering all paragraphs in the Pad.
- Superparagraph average length: Average length in characters considering all superparagraphs in the Pad.
- Average paragraphs/superparagraph: Average number of paragraphs per superparagraph considering all paragraphs and superparagraphs in the Pad.

Some of these measures can be computed over a time window, as shown in the following table:

| Heuristic name                   | Window |
|----------------------------------|--------|
| Alternating                      | No     |
| Day break                        | Yes    |
| Short break                      | Yes    |
| Overall type                     | Yes    |
| Proportion                       | No     |
| Synchronous                      | Yes    |
| User type                        | Yes    |
| User proportion per paragraph    | No     |
| Length                           | No     |
| Length all                       | Yes    |
| Length all write                 | Yes    |
| Length all paste                 | Yes    |
| Added characters                 | Yes    |
| Deleted characters               | Yes    |
| Paragraph average length         | No     |
| Superparagraph average length    | No     |
| Average paragraphs/superparagraph| No     |


### Command line execution
We have provided a small command line interface to interact with the program.
```
python run_analytics.py
```
With the following arguments:
```
usage: run_analytics.py [-h] [-p PATH_TO_DB]
                        [-e {etherpad,etherpadSQLite3,stian_logs,collab-react-components,sql_dump,MySQL}]
                        [-t] [--generate_csv] [--generate_csv_summary]
                        [--start_time START_TIME] [-viz] [-v]
                        [--subset_of_pads SUBSET_OF_PADS | --specific_pad SPECIFIC_PAD]

Run the analytics.

optional arguments:
  -h, --help            show this help message and exit
  -p PATH_TO_DB, --path_to_db PATH_TO_DB
                        Path to the database storing the edits
  -e {etherpad,etherpadSQLite3,stian_logs,collab-react-components,sql_dump,MySQL}, --editor {etherpad,etherpadSQLite3,stian_logs,collab-react-components,sql_dump,MySQL}
                        Editor where the logs are from
  -t, --texts           Print the texts colored by ops and by author
  --generate_csv        Generate a csv file containing the information for all
                        pads
  --generate_csv_summary
                        Generate a csv file containing the summarized
                        information for all pads
  --start_time START_TIME
                        Define the start time for the time window to generate
                        the csv summary
  -viz, --visualization
                        Display the visualization
  -v, --verbosity       Increase output verbosity (either -v or -vv)
  --subset_of_pads SUBSET_OF_PADS
                        Size of the subset of pads to be processed
  --specific_pad SPECIFIC_PAD
                        Specify the name of a single pad to be processed

 ```

Below are a few examples of execution. In some cases only one of the options is considered, we suggest looking at the documentation in the code for more details.

#### Examples of execution
##### Etherpad
To launch the program on data collected with the Etherpad editor. You can launch the following command:

```
python run_analytics.py -p "etherpad/dirty.db" -e etherpad -t -viz -v --specific_pad "First Pad"
```
This will extract the logs in `etherpad/dirty.db` and provide a few insights on the pad "First Pad". You could also remove `--specific_pad "First Pad"` so that you display the insights for all pads in the logs. The `-t` argument will print the text colored by authors and by operations. The `-viz` argument will display the various visualizations explained later.
> Note : it is possible that your terminal doesn't handle colors. In this case, you will need to run it in a python console or remove the -t argument.

##### Collab-React-Components

If you would like to run the program on all the pads of the Collab-React-Components editor, then you can run the following command
```
python run_analytics.py -e collab-react-components -t -viz -v
```
> Note: there is no need to specify the path of the logs since they are store in the mongo database.

### Live analytics for Etherpad and collab-react-components

Live analytics updates the metrics, as soon as there is a change in the document. You can run it with:
```
python live_analytics.py
```
> Note: live_analytics.py doesn't take any command line arguments. We suggest you configure which editor you are using in config.py !

### Live analytics for FROG

Live analytics for FROG receives by HTTP/POST in a json the pad_names it wants to follow. It can also send a regex. Then every few seconds (the refresh rate is configurable in the config.py file), the program sends a HTTP/POST to a listening server (adress specified in the config.py).
You can run the server with :

```
export FLASK_APP=server.py
flask run --host=0.0.0.0
```

The HTTP/POST defining the pad_names we want to follow should be of the following format :

```
{'pad_names': ["/ac-textarea/default/0","/ac-textarea/default/0"]}
```

or we can send a regex

```
{'regex': "^/ac-textarea/default"}
```

Finally, by sending a HTTP/POST without a json, we will get updates on all the existing pads at the time of the POST.

The answer will be of the following format:
```
{<pad name>: {'Alternating score:': 0.75,
              'Break score day:': 4.286906128731559e-06,
              'Break score short:': 0.0,
              'Overall delete type score:': 0.043478260869565216,
              'Overall edit type score:': 0.6086956521739131,
              'Overall paste type score:': 0.0,
              'Overall write type score:': 0.34782608695652173,
              'Proportion score:': 0.7069812420203414,
              'Synchronous score:': 0.9533678756476682,
              'User delete score:': 3.855291626815406e-05,
              'User edit score:': 0.860781638418886,
              'User paste score:': 4.626349952178488e-05,
              'User proportion per paragraph score': 3.855291626815406e-05,
              'User write score:': 0.8015329750891311,
              'text': <document text>,
              'text_colored_by_authors': <text colored by authors>,
              'text_colored_by_ops': <text colored by ops>}
```

### Save a tsv file containing measures for all pads
```
python run_analytics.py -p my_data.sql -e sql_dump --generate_csv_summary --start_time 0 > results_file.tsv
```

To see how to generate the measures within specified time intervals, you can see a more detailed example in file `src/analysis_evolution.ipynb`.

### Save a tsv file containing measures for each operation in all pads
```
python run_analytics.py -p my_data.sql -e sql_dump --generate_csv > results_file.tsv
```

### Architecture
The whole program use various files:
- analytics directory:
  - Operations.py: This file contains the classes of ElementaryOperation and Operations. An ElementaryOperation is mostly defined by its position, its type (add or del), its text to add or length to delete, timestamp and author. It also contains additional attributes used on the backend such as its position at a certain time, the Operation it belongs to and so on... An Operation is a list of ElementaryOperation. It is mostly defined by its start position, its length, starting and ending timestamp, author, its type, its context and the list of ElementaryOperation it is composed of. The list of ElementaryOperation must be of the same author, written without pausing for too long and intersecting each other. The Operation's type is computed by studying this list and classifying the Operation in either write (writing a big amount of letters), edit (writing/deleting/replacing a small amount of letters), delete (deleting a big amount of letters), paste (from copy/paste) and jump (creating a new line. When a user adds a new line, it is considered as a new operation). The operation context contains information such as whether this operation was written synchronously with other authors, how big it is compared to the rest of the operations and so on...
  - operation_builder.py: Contains the methods to cluster the ElementaryOperation into Operations and creating the corresponding Pads.
  - Pad.py: Defines the class Pad. A Pad is a list of operations and a list of paragraph. It corresponds to a document. It is from a Pad we will calculate the metrics. A paragraph is the list of operation that are, at this state of the pad, on the same line. This allows us to create metrics such as studying whether two users worked together on a smaller scale on the same paragraph or whether they each wrote their own block of text. The metrics calculated from a pad are explained on the section [Usage](#usage)
  - parser.py: This file contains all the methods used to extract the ElementaryOperation from the database of the editor.
  - visualization.py: Contains the code used to create various visualizations. It is used by most of the main files. It contains visualization to show the participation of each user, their writing style and so on...
- config.py: This file contains all the tweakable parameters. There is a description of each paramater in the file. You can configure the editor type, the path to the database, if applicable, the various parameters impacting the operation computations and the mongo database connection information, if applicable.
- main files: Files we used to test and develop the application. It might be necessary to modify them in order to use them. `main.py` was used to calculate the metrics and create the visualization for documents written in etherpad. `main_stian_logs.py` was used to calculate the metrics and create the visualization for Stian's data. `main_belgians.py` and `main_sydney.py` were used to calculate the metrics and create the visualization on the documents from the Belgian and Sydney experiments respectively.  `main_belgians_evolution.py` was used to study the evolution of the metrics on the documents from the Belgian experiment.
- run_analytics.py: It is a python file that can be run with various parameters. See [Command line execution](#command-line-execution). The parameters override those written in config.py.
- live_analytics.py: It is a runnable python file than can display the metrics live for documents. See [Live analytics for Etherpad and collab-react-components](#live-analytics-for-etherpad-and-collab-react-components)
- server.py: As explained in [Live analytics for FROG](#live-analytics-for-frog), it creates a webserver that listen for HTTP/POST requests that explains what we want to listen in a json payload. It then creates two threads. One of them parse all the writing events from the editor's database and then look for new unprocessed writing events at a certain refresh rate defined in config.py. The other thread sends, at a certain refresh rate defined in config.py, the metrics to a url defined in config.py in a HTTP/POST requests as a json payload.
- Notebooks: The notebooks contain small analysis and examples done on different data.


### Visualization
> Note: The implementation of the visualization methods can be found in `visualization.py`

Show the text with the different authors using `display_text_colored_by_authors`, here is an example with the admin and two other authors:
![](readme_figures/authors.JPG)

Show the same text with `Operations` randomly colored using `display_text_colored_by_ops:
![](readme_figures/operations.JPG)

Show the overall proportion of participation of the pad using `display_user_participation`:
> Note: We consider participations to be absolute. So if a user delete for example a line, it counts as a participation. See below for a separated visualization.

![](readme_figures/Demo_user_participation.png)

Show the same proportion as before but for each `Paragraphs` using `display_user_participation_paragraphs`:
![](readme_figures/Demo_user_abs_participation_para.png)

Show the same proportion as before but with addition and deletions separated `display_user_participation_paragraphs_with_del`:
![](readme_figures/Demo_user_participation_para.png)

Show the proportion of the pad written synchronously using `display_proportion_sync_in_pad`:
> Note: We don't take into account the admin here whereas in the next figure we do in order to be consistent with the previous figures.

![](readme_figures/Demo_sync_prop_pad.png)

Show the proportion written synchronously of each `Paragraphs` using `display_proportion_sync_in_paragraphs`:
![](readme_figures/Demo_sync_prop_para.png)

Show the distribution of `Operation` different types (except Jump) in one pad using `display_overall_op_type`:
<img src="readme_figures/Demo_overall_op_type.png" height="700" width="800">

Show the same as above but according to authors using `display_types_per_user`:
<img src="readme_figures/Demo_types_per_user.png" height="700" width="800">


### Future work
- Implement visualization for the Breaks measures.
- Tuning the parameters of the classification of the operations (write, edit, delete, paste) - it depends on the data.
- Creating new operations that are more relevant?
- Redefining paragraph identifiers so that they do not become as long, also so that they are more intuitive and easier to analyze.
- Improving the way we define time windows so that it is not for a constant length, but it depends on the total length on the Pad - useful for comparing pads that do not last for exactly the same time.
- Fixing technical issues.
- Finding a metric to encapsulate the collaboration of the students based on the operation types and the context.
- Identifying how useful the different metrics and Operation context information is, as there is quite a lot of information right now and not all of it might be equally useful/relevant.
- Optimize it. A lot. (Rewrite some critical parts of the code, parallelization...)