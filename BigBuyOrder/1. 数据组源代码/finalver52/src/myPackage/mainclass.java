
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
import java.lang.Math;
import java.util.ArrayList;

public class mainclass {
    //main function
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://hadoop-namenode:9082");
        conf.set("mapreduce.framework.name", "yarn");
        conf.set("yarn.nodemanager.aux-services", "mapreduce_shuffle");
        conf.set("yarn.resourcemanager.hostname", "hadoop-namenode");
        conf.set("mapreduce.jobhistory.address","hadoop-namenode:10020");

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
            String[] info = line.split("\t");

            String stock_time = info[0].substring(0, 15);

            context.write(new Text(stock_time), new Text(info[1]));
        }
    }


    public static class New_Reducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            FloatWritable[] slice_data = new FloatWritable[3];

            float cum_amount = 0;
            float cum_volume = 0;
            float mean_volume = 0;
            float var_volume = 0;
            float std_volume = 0;
            float number = 0;
            float big_buy = 0;
            float big_sell = 0;
            Text output = new Text();
            ArrayList<String> arrayList = new ArrayList<String>();

            for (Text value : values) {
                String line = value.toString();
                arrayList.add(line);
                String[] info = line.split(",");
                float this_volume = Float.parseFloat(info[1]);
                float this_amount = Float.parseFloat(info[2]);
                number += 1;
                cum_volume += this_volume;
                cum_amount += this_amount;
            }
            mean_volume = cum_volume / number;

            for (String value : arrayList) {
                String line = value.toString();
                String[] info = line.split(",");
                float this_volume = Float.parseFloat(info[1]);
                var_volume += (this_volume - mean_volume) * (this_volume - mean_volume);
            }
            std_volume = (float) Math.sqrt(var_volume/number);

            for (String value : arrayList) {
                String line = value.toString();
                String[] info = line.split(",");
                float this_volume = Float.parseFloat(info[1]);
                float this_amount = Float.parseFloat(info[2]);

                if (this_volume > (mean_volume + std_volume)) {
                    if (info[3].equals("s")) {
                        big_sell += this_amount;
                    } else {
                        big_buy += this_amount;
                    }
                }
            }


            slice_data[0] = new FloatWritable(cum_amount);
            slice_data[2] = new FloatWritable(big_buy / cum_amount);
            slice_data[1] = new FloatWritable(big_sell / cum_amount);

            output.set(slice_data[0].toString() + "," + slice_data[1].toString() + "," +
                    slice_data[2].toString());

            context.write(key, output);
        }
    }
}

