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
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://hadoop-namenode:9082");
        conf.set("mapreduce.framework.name", "yarn");
        conf.set("yarn.nodemanager.aux-services", "mapreduce_shuffle");
        conf.set("yarn.resourcemanager.hostname", "hadoop-namenode");
        conf.set("mapreduce.jobhistory.address", "hadoop-namenode:10020");

        Job job = Job.getInstance(conf, "fianl_project");
        job.setJarByClass(mainclass.class);

        job.setMapperClass(New_Mapper.class);
        job.setReducerClass(New_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.out.println(job.waitForCompletion(true) ? 0 : 1);
    }


    public static class New_Mapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();
            String[] info = line.split(",");

            if(info[1].substring(8, 12).compareTo("1459")>=0){
                    return;
            }
            String stock_time = info[0] + "_" + info[1].substring(0, 8);

            String close_price = info[2];
            String volume = info[3];
            String amount = info[4];
            String buy_side = info[5];
            String sell_side = info[6];

            context.write(new Text(stock_time + "_" + buy_side), new Text(close_price + ","
                    + volume + "," + amount + ",b"));
            context.write(new Text(stock_time + "_" + sell_side), new Text(close_price + ","
                    + volume + "," + amount + ",s"));
        }
    }


    public static class New_Reducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            FloatWritable[] slice_data = new FloatWritable[3];
            float volume = 0;
            float close = 0;
            float amount = 0;
            String status = "";
            Text output = new Text();

            for (Text value : values) {
                String line = value.toString();
                String[] info = line.split(",");
                float this_volume = Float.parseFloat(info[1]);
                float this_amount = Float.parseFloat(info[2]);

                volume += this_volume;
                amount += this_amount;
                status = info[3];
            }

            slice_data[0] = new FloatWritable(amount / volume);
            slice_data[1] = new FloatWritable(volume);
            slice_data[2] = new FloatWritable(amount);


            output.set(slice_data[0].toString() + "," + slice_data[1].toString() + "," +
                    slice_data[2].toString() + "," + status);

            context.write(key, output);
        }
    }
}



