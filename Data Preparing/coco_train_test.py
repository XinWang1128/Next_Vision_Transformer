<html>
<head>
<title>coco_train_test.py</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<style type="text/css">
.s0 { color: #808080;}
.s1 { color: #a9b7c6;}
.s2 { color: #629755; font-style: italic;}
.s3 { color: #cc7832;}
.s4 { color: #6a8759;}
.s5 { color: #6897bb;}
</style>
</head>
<body bgcolor="#2b2b2b">
<table CELLSPACING=0 CELLPADDING=5 COLS=1 WIDTH="100%" BGCOLOR="#606060" >
<tr><td><center>
<font face="Arial, Helvetica" color="#000000">
coco_train_test.py</font>
</center></td></tr></table>
<pre><span class="s0"># -------------------------------</span>
<span class="s0"># author: Hao Li, hao.li@uni-heidelberg.de</span>
<span class="s0"># data: 06.03.2021</span>
<span class="s0"># -------------------------------</span>

<span class="s2">r&quot;&quot;&quot;Convert raw Microsoft COCO dataset to TFRecord for object_detection. 
 
1) Installation: 
    pip install pycocotools-windows 
 
2) For easy use of this script, Your coco dataset directory structure should like this : 
    +Your coco dataset root 
        +image 
        +annotation 
            -train.json 
            -test.json 
 
 
Example usage: 
    python tf_record_from_coco.py --label_input= ./coco_repo 
             --train_rd_path=data/train_xxx.record \ 
             --valid_rd_path=data/valid_xxx.record 
 
    python tf_record_from_coco.py --label_input=C:\Users\24634\PycharmProjects\pythonProject\ohsome2label\STUT_WWTP --train_rd_path=BW_WWTP\train.record --valid_rd_path=BW_WWTP\valid.record 
 
&quot;&quot;&quot;</span>

<span class="s3">from </span><span class="s1">__future__ </span><span class="s3">import </span><span class="s1">absolute_import</span>
<span class="s3">from </span><span class="s1">__future__ </span><span class="s3">import </span><span class="s1">division</span>
<span class="s3">from </span><span class="s1">__future__ </span><span class="s3">import </span><span class="s1">absolute_import</span>
<span class="s3">from </span><span class="s1">__future__ </span><span class="s3">import </span><span class="s1">division</span>
<span class="s3">from </span><span class="s1">__future__ </span><span class="s3">import </span><span class="s1">print_function</span>
<span class="s3">from </span><span class="s1">pycocotools.coco </span><span class="s3">import </span><span class="s1">COCO</span>
<span class="s3">from </span><span class="s1">PIL </span><span class="s3">import </span><span class="s1">Image</span>
<span class="s3">from </span><span class="s1">random </span><span class="s3">import </span><span class="s1">shuffle</span>
<span class="s3">import </span><span class="s1">os</span><span class="s3">, </span><span class="s1">sys</span>
<span class="s3">import </span><span class="s1">shutil</span>
<span class="s3">import </span><span class="s1">numpy </span><span class="s3">as </span><span class="s1">np</span>
<span class="s3">import </span><span class="s1">tensorflow.compat.v1 </span><span class="s3">as </span><span class="s1">tf</span>
<span class="s3">import </span><span class="s1">logging</span>
<span class="s3">from </span><span class="s1">os </span><span class="s3">import </span><span class="s1">makedirs</span>
<span class="s3">import </span><span class="s1">shutil</span>

<span class="s3">import </span><span class="s1">dataset_util</span>

<span class="s1">flags = tf.app.flags</span>
<span class="s1">flags.DEFINE_string(</span><span class="s4">'label_input'</span><span class="s3">, </span><span class="s4">''</span><span class="s3">, </span><span class="s4">'Root directory to image and annotation.'</span><span class="s1">)</span>
<span class="s1">flags.DEFINE_string(</span><span class="s4">'train_rd_path'</span><span class="s3">, </span><span class="s4">''</span><span class="s3">, </span><span class="s4">'Path to output train TFRecord'</span><span class="s1">)</span>
<span class="s1">flags.DEFINE_string(</span><span class="s4">'valid_rd_path'</span><span class="s3">, </span><span class="s4">''</span><span class="s3">, </span><span class="s4">'Path to output validate TFRecord'</span><span class="s1">)</span>
<span class="s1">FLAGS = flags.FLAGS</span>


<span class="s3">def </span><span class="s1">clean_image(imgs_dir</span><span class="s3">, </span><span class="s1">preview_dir</span><span class="s3">, </span><span class="s1">annotations_filepath):</span>
    <span class="s2">&quot;&quot;&quot;Clean image sets based on preview results. 
    Args: 
        imgs_dir: directories of coco images 
        preview_dir: preselection of images samples 
        annotations_filepath: file path of coco annotations file 
    Return: 
        no reture 
    &quot;&quot;&quot;</span>
    <span class="s1">coco = COCO(annotations_filepath)</span>
    <span class="s1">img_ids = coco.getImgIds()</span>
    <span class="s1">cat_ids = coco.getCatIds()</span>

    <span class="s3">for </span><span class="s1">index</span><span class="s3">, </span><span class="s1">img_id </span><span class="s3">in </span><span class="s1">enumerate(img_ids):</span>
        <span class="s1">img_detail = coco.loadImgs(img_id)[</span><span class="s5">0</span><span class="s1">]</span>
        <span class="s1">img_path = os.path.join(imgs_dir</span><span class="s3">, </span><span class="s1">img_detail[</span><span class="s4">'file_name'</span><span class="s1">])</span>
        <span class="s1">preview_path = os.path.join(preview_dir</span><span class="s3">, </span><span class="s1">img_detail[</span><span class="s4">'file_name'</span><span class="s1">])</span>
        <span class="s3">if not </span><span class="s1">os.path.isfile(preview_path):</span>
            <span class="s1">os.remove(img_path)</span>


<span class="s3">def </span><span class="s1">load_coco_dection_dataset(imgs_dir</span><span class="s3">, </span><span class="s1">annotations_filepath):</span>
    <span class="s2">&quot;&quot;&quot;Load data from dataset by pycocotools. This tools can be download from &quot;http://mscoco.org/dataset/#download&quot; 
    Args: 
        imgs_dir: directories of coco images 
        annotations_filepath: file path of coco annotations file 
    Return: 
        coco_data: list of dictionary format information of each image 
    &quot;&quot;&quot;</span>
    <span class="s1">coco = COCO(annotations_filepath)</span>
    <span class="s1">img_ids = coco.getImgIds()</span>
    <span class="s1">cat_ids = coco.getCatIds()</span>

    <span class="s0"># if shuffle_img:</span>
    <span class="s0">#     shuffle(img_ids)</span>

    <span class="s1">coco_data = []</span>

    <span class="s1">nb_imgs = len(img_ids)</span>
    <span class="s3">for </span><span class="s1">index</span><span class="s3">, </span><span class="s1">img_id </span><span class="s3">in </span><span class="s1">enumerate(img_ids):</span>
        <span class="s1">img_info = {}</span>
        <span class="s1">bboxes = []</span>
        <span class="s1">labels = []</span>
        <span class="s1">entity = []</span>

        <span class="s1">img_detail = coco.loadImgs(img_id)[</span><span class="s5">0</span><span class="s1">]</span>
        <span class="s1">pic_height = img_detail[</span><span class="s4">'height'</span><span class="s1">]</span>
        <span class="s1">pic_width = img_detail[</span><span class="s4">'width'</span><span class="s1">]</span>

        <span class="s1">ann_ids = coco.getAnnIds(imgIds=img_id</span><span class="s3">, </span><span class="s1">catIds=cat_ids)</span>
        <span class="s1">anns = coco.loadAnns(ann_ids)</span>
        <span class="s3">for </span><span class="s1">ann </span><span class="s3">in </span><span class="s1">anns:</span>
            <span class="s1">bboxes_data = ann[</span><span class="s4">'bbox'</span><span class="s1">]</span>
            <span class="s1">cats = coco.loadCats(ann[</span><span class="s4">'category_id'</span><span class="s1">])[</span><span class="s5">0</span><span class="s1">][</span><span class="s4">&quot;name&quot;</span><span class="s1">]</span>
            <span class="s1">bboxes_data = [bboxes_data[</span><span class="s5">0</span><span class="s1">] / float(pic_width)</span><span class="s3">, </span><span class="s1">bboxes_data[</span><span class="s5">1</span><span class="s1">] / float(pic_height)</span><span class="s3">, </span><span class="s1">\</span>
                           <span class="s1">bboxes_data[</span><span class="s5">2</span><span class="s1">] / float(pic_width)</span><span class="s3">, </span><span class="s1">bboxes_data[</span><span class="s5">3</span><span class="s1">] / float(pic_height)]</span>
            <span class="s0"># the format of coco bounding boxs is [Xmin, Ymin, width, height]</span>
            <span class="s1">bboxes.append(bboxes_data)</span>
            <span class="s1">labels.append(ann[</span><span class="s4">'category_id'</span><span class="s1">])</span>
            <span class="s1">entity.append(cats.encode(</span><span class="s4">'utf8'</span><span class="s1">))</span>

        <span class="s1">img_path = os.path.join(imgs_dir</span><span class="s3">, </span><span class="s1">img_detail[</span><span class="s4">'file_name'</span><span class="s1">])</span>
        <span class="s0">#preview_path = os.path.join(preview_dir, img_detail['file_name'])</span>
        <span class="s3">if </span><span class="s1">os.path.isfile(img_path):</span>
            <span class="s1">img_bytes = tf.gfile.FastGFile(img_path</span><span class="s3">, </span><span class="s4">'rb'</span><span class="s1">).read()</span>
            <span class="s1">img_info[</span><span class="s4">'pixel_data'</span><span class="s1">] = img_bytes</span>
            <span class="s1">img_info[</span><span class="s4">'height'</span><span class="s1">] = pic_height</span>
            <span class="s1">img_info[</span><span class="s4">'width'</span><span class="s1">] = pic_width</span>
            <span class="s1">img_info[</span><span class="s4">'bboxes'</span><span class="s1">] = bboxes</span>
            <span class="s1">img_info[</span><span class="s4">'labels'</span><span class="s1">] = labels</span>
            <span class="s1">img_info[</span><span class="s4">'text'</span><span class="s1">] = entity</span>
            <span class="s1">img_info[</span><span class="s4">'file'</span><span class="s1">] = img_detail[</span><span class="s4">'file_name'</span><span class="s1">]</span>
            <span class="s1">coco_data.append(img_info)</span>
    <span class="s3">return </span><span class="s1">coco_data</span>


<span class="s3">def </span><span class="s1">dict_to_coco_example(img_data):</span>
    <span class="s2">&quot;&quot;&quot;Convert python dictionary formath data of one image to tf.Example proto. 
    Args: 
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\ 
            height, width, encoded pixel data. 
    Returns: 
        example: The converted tf.Example 
    &quot;&quot;&quot;</span>
    <span class="s1">bboxes = img_data[</span><span class="s4">'bboxes'</span><span class="s1">]</span>
    <span class="s1">xmin</span><span class="s3">, </span><span class="s1">xmax</span><span class="s3">, </span><span class="s1">ymin</span><span class="s3">, </span><span class="s1">ymax = []</span><span class="s3">, </span><span class="s1">[]</span><span class="s3">, </span><span class="s1">[]</span><span class="s3">, </span><span class="s1">[]</span>
    <span class="s3">for </span><span class="s1">bbox </span><span class="s3">in </span><span class="s1">bboxes:</span>
        <span class="s1">xmin.append(bbox[</span><span class="s5">2</span><span class="s1">])</span>
        <span class="s1">xmax.append(bbox[</span><span class="s5">0</span><span class="s1">])</span>
        <span class="s1">ymin.append(bbox[</span><span class="s5">3</span><span class="s1">])</span>
        <span class="s1">ymax.append(bbox[</span><span class="s5">1</span><span class="s1">])</span>
    <span class="s1">example = tf.train.Example(features=tf.train.Features(feature={</span>
        <span class="s4">'image/height'</span><span class="s1">: dataset_util.int64_feature(img_data[</span><span class="s4">'height'</span><span class="s1">])</span><span class="s3">,</span>
        <span class="s4">'image/width'</span><span class="s1">: dataset_util.int64_feature(img_data[</span><span class="s4">'width'</span><span class="s1">])</span><span class="s3">,</span>
        <span class="s4">'image/object/bbox/xmin'</span><span class="s1">: dataset_util.float_list_feature(xmin)</span><span class="s3">,</span>
        <span class="s4">'image/object/bbox/xmax'</span><span class="s1">: dataset_util.float_list_feature(xmax)</span><span class="s3">,</span>
        <span class="s4">'image/object/bbox/ymin'</span><span class="s1">: dataset_util.float_list_feature(ymin)</span><span class="s3">,</span>
        <span class="s4">'image/object/bbox/ymax'</span><span class="s1">: dataset_util.float_list_feature(ymax)</span><span class="s3">,</span>
        <span class="s4">'image/object/class/label'</span><span class="s1">: dataset_util.int64_list_feature(img_data[</span><span class="s4">'labels'</span><span class="s1">])</span><span class="s3">,</span>
        <span class="s4">'image/object/class/text'</span><span class="s1">: dataset_util.bytes_list_feature(img_data[</span><span class="s4">'text'</span><span class="s1">])</span><span class="s3">,</span>
        <span class="s4">'image/encoded'</span><span class="s1">: dataset_util.bytes_feature(img_data[</span><span class="s4">'pixel_data'</span><span class="s1">])</span><span class="s3">,</span>
        <span class="s4">'image/format'</span><span class="s1">: dataset_util.bytes_feature(</span><span class="s4">'jpeg'</span><span class="s1">.encode(</span><span class="s4">'utf-8'</span><span class="s1">))</span><span class="s3">,</span>
        <span class="s4">'image/object/class/file'</span><span class="s1">: dataset_util.bytes_feature(img_data[</span><span class="s4">'file'</span><span class="s1">].encode(</span><span class="s4">'utf-8'</span><span class="s1">))</span><span class="s3">,</span>
    <span class="s1">}))</span>
    <span class="s3">return </span><span class="s1">example</span>

<span class="s3">def </span><span class="s1">main(_):</span>
    <span class="s1">imgs_dir = os.path.join(FLAGS.label_input</span><span class="s3">, </span><span class="s4">'images'</span><span class="s1">)</span>
    <span class="s1">preview_dir = os.path.join(FLAGS.label_input</span><span class="s3">, </span><span class="s4">'preview'</span><span class="s1">)</span>
    <span class="s1">annotations_filepath = os.path.join(FLAGS.label_input</span><span class="s3">, </span><span class="s4">'annotations'</span><span class="s3">, </span><span class="s4">'geococo.json'</span><span class="s1">)</span>
    <span class="s1">print(</span><span class="s4">&quot;Convert coco val file to tf record&quot;</span><span class="s1">)</span>
    <span class="s1">coco_data = load_coco_dection_dataset(imgs_dir</span><span class="s3">, </span><span class="s1">annotations_filepath)</span>
    <span class="s0">#clean_image(imgs_dir, preview_dir,</span>
    <span class="s0">#            annotations_filepath)  # optional funtion of delecting bad samples in the preview folder</span>
    <span class="s1">total_imgs = len(coco_data)</span>
    <span class="s0"># print(coco_data)</span>
    <span class="s0"># breakpoint() # insert breakpoint</span>

    <span class="s0"># define the first 80% of the input data as training data</span>
    <span class="s1">split_index = int(total_imgs * </span><span class="s5">0.8</span><span class="s1">)</span>
    <span class="s1">coco_data_train = coco_data[:split_index]</span>

    <span class="s0"># define the another 20% of the input as test data</span>
    <span class="s1">coco_data_validation = coco_data[split_index:]</span>
    <span class="s1">train_dir = os.path.join(FLAGS.label_input</span><span class="s3">, </span><span class="s4">'train'</span><span class="s1">)</span>
    <span class="s1">test_dir = os.path.join(FLAGS.label_input</span><span class="s3">, </span><span class="s4">'test'</span><span class="s1">)</span>

   <span class="s0"># create the path of train and test</span>
    <span class="s3">if not </span><span class="s1">os.path.isdir(train_dir):</span>
        <span class="s1">makedirs(train_dir)</span>
    <span class="s3">if not </span><span class="s1">os.path.isdir(test_dir):</span>
        <span class="s1">makedirs(test_dir)</span>

    <span class="s0">#</span>
    <span class="s3">for </span><span class="s1">train_tile </span><span class="s3">in </span><span class="s1">coco_data_train:</span>
        <span class="s1">file = train_tile[</span><span class="s4">'file'</span><span class="s1">]</span>
        <span class="s1">tile_dir = os.path.join(imgs_dir</span><span class="s3">, </span><span class="s1">file)</span>
        <span class="s1">shutil.copy(tile_dir</span><span class="s3">, </span><span class="s1">train_dir)</span>

    <span class="s3">for </span><span class="s1">valid_tile </span><span class="s3">in </span><span class="s1">coco_data_validation:</span>
        <span class="s1">file = valid_tile[</span><span class="s4">'file'</span><span class="s1">]</span>
        <span class="s1">tile_dir = os.path.join(imgs_dir</span><span class="s3">, </span><span class="s1">file)</span>
        <span class="s1">shutil.copy(tile_dir</span><span class="s3">, </span><span class="s1">test_dir)</span>

    <span class="s0"># create the  train.json and test.json</span>
    <span class="s3">with </span><span class="s1">tf.python_io.TFRecordWriter(FLAGS.train_rd_path) </span><span class="s3">as </span><span class="s1">tfrecord_writer:</span>
        <span class="s3">for </span><span class="s1">index</span><span class="s3">, </span><span class="s1">img_data </span><span class="s3">in </span><span class="s1">enumerate(coco_data_train):</span>
            <span class="s1">example = dict_to_coco_example(img_data)</span>
            <span class="s1">tfrecord_writer.write(example.SerializeToString())</span>
            <span class="s1">train_annotations_filepath = os.path.join(example</span><span class="s3">, </span><span class="s4">'annotations'</span><span class="s3">, </span><span class="s4">'traincoco.json'</span><span class="s1">)</span>
        <span class="s1">print(</span><span class="s4">&quot;Converted in total {} images for training!&quot;</span><span class="s1">.format(len(coco_data_train)))</span>


    <span class="s3">with </span><span class="s1">tf.python_io.TFRecordWriter(FLAGS.valid_rd_path) </span><span class="s3">as </span><span class="s1">tfrecord_writer:</span>
        <span class="s3">for </span><span class="s1">index</span><span class="s3">, </span><span class="s1">img_data </span><span class="s3">in </span><span class="s1">enumerate(coco_data_validation):</span>
            <span class="s1">example = dict_to_coco_example(img_data)</span>
            <span class="s1">tfrecord_writer.write(example.SerializeToString())</span>
            <span class="s1">test_annotations_filepath = os.path.join(example</span><span class="s3">, </span><span class="s4">'annotations'</span><span class="s3">, </span><span class="s4">'testcoco.json'</span><span class="s1">)</span>
        <span class="s1">print(</span><span class="s4">&quot;Converted in total {} images for validating!&quot;</span><span class="s1">.format(len(coco_data_validation)))</span>


<span class="s3">if </span><span class="s1">__name__ == </span><span class="s4">&quot;__main__&quot;</span><span class="s1">:</span>
    <span class="s1">tf.app.run()</span>
</pre>
</body>
</html>