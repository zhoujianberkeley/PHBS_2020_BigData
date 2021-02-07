package myPackage;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;


public class mainclass {
    //main function
    public static void main(String[] args) throws Exception {
        Configuration conf =new Configuration();
        conf.set("fs.defaultFS", "hdfs://hadoop-namenode:9082");
        conf.set("mapreduce.framework.name", "yarn");
        conf.set("yarn.nodemanager.aux-services", "mapreduce_shuffle");
        conf.set("yarn.resourcemanager.hostname", "hadoop-namenode");
        conf.set("mapreduce.jobhistory.address","hadoop-namenode:10020");

        Job job = Job.getInstance(conf,"fianl_project");
        job.setJarByClass(mainclass.class);

        job.setMapperClass(New_Mapper.class);
        job.setReducerClass(New_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.out.println(job.waitForCompletion(true) ? 0 : 1);
    }

    // 第一种切片方式
    public static class New_Mapper extends Mapper <LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context )
                throws IOException, InterruptedException{

            String line = value.toString();
            String[] info = line.split(",");

            String stock_time = info[0]+"_"+info[1].substring(0,12);

            String closePrice = info[2];
            String volume = info[3];

            context.write(new Text(stock_time), new Text(closePrice+","+volume));

        }
    }



    public static class New_Reducer extends Reducer<Text, Text, Text, Text>{
//        private MultipleOutputs<Text, Text> multipleOutputs;
//
//        protected void setup(Context context) {
//            multipleOutputs = new MultipleOutputs<Text, Text>(context);
//        }

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            FloatWritable[] slice_data = new FloatWritable[6];
            Text output = new Text();

            float open_price = 0;
            float high_price = 0;
            float low_price = 0;
            float close_price = 0;
            float volume = 0;
            float amount = 0;

            for (Text value : values) {
                String line = value.toString();
                String[] info = line.split(",");
                float close = Float.parseFloat(info[0]);
                float this_volume = Float.parseFloat(info[1]);

                if (open_price==0) {
                    open_price = close;
                }

                if (low_price==0){
                    low_price = close;
                }

                high_price = Math.max(high_price, close);
                low_price = Math.min(low_price, close);
                close_price = close;

                volume += this_volume;
                amount = this_volume*close;
            }

            slice_data[0] = new FloatWritable(open_price);
            slice_data[1] = new FloatWritable(high_price);
            slice_data[2] = new FloatWritable(low_price);
            slice_data[3] = new FloatWritable(close_price);
            slice_data[4] = new FloatWritable(volume);
            slice_data[5] = new FloatWritable(amount);


            output.set(slice_data[0].toString()+","+slice_data[1].toString()+","+slice_data[2].toString()+","+
                    slice_data[3].toString()+","+slice_data[4].toString()+","+slice_data[5].toString());
            //context.write(key, new FloatArrayWritable(output.get()));
            context.write(key, output);


        }



//        protected void cleanup(Context context) throws IOException, InterruptedException{
//            multipleOutputs.close();
//        }

    }


}
