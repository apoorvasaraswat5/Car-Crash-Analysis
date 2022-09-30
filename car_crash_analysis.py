import os
from configparser import RawConfigParser

from pyspark.sql.utils import AnalysisException, IllegalArgumentException
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lit, sum, count, row_number
from pyspark.sql import SparkSession


class CarCrashAnalysis(object):
    """
    This class reads the files present in the folder Data_files
    which contains files related to Car Crashes in US.
    The method execute_analysis performs various analysis using
    the provided data.
    The output of the analysis is written in the file path provided
    in Configurations/Configurations.conf
    """

    def __init__(self):
        self._config_reader = RawConfigParser()
        self._config_reader.read("Configuration.conf")
        self._input_file_path = self._config_reader.get("INPUT", "FILE_PATH")
        self._output_file_path = self._config_reader.get("OUTPUT", "FILE_PATH")

        self.spark_session = self.get_spark_session()
        self.charges_use_df = self.get_dataframe_from_file_path(self._get_file_path("Charges_use.csv"))
        self.damages_use_df = self.get_dataframe_from_file_path(self._get_file_path("Damages_use.csv"))
        self.endorse_use_df = self.get_dataframe_from_file_path(self._get_file_path("Endorse_use.csv"))
        self.primary_person_use_df = self.get_dataframe_from_file_path(self._get_file_path("Primary_Person_use.csv"))
        self.restrict_use_df = self.get_dataframe_from_file_path(self._get_file_path("Restrict_use.csv"))
        self.units_use_df = self.get_dataframe_from_file_path(self._get_file_path("Units_use.csv"))
        self._output = ""

    @staticmethod
    def get_spark_session():
        spark_session = SparkSession.builder.enableHiveSupport().appName('myApp').getOrCreate()
        return spark_session

    def _get_file_path(self, file_name):
        return self._input_file_path.format(file_name=file_name)

    def get_dataframe_from_file_path(self, path):
        df = self.spark_session.read.csv(path=path, header=True)
        return df

    def get_no_of_car_crashes_persons_killed_male(self):
        """
        Find the number of crashes (accidents) in which number of persons killed are male
        """
        result = self.primary_person_use_df.filter((self.primary_person_use_df.DEATH_CNT > 0) & (
                self.primary_person_use_df.PRSN_GNDR_ID == "MALE")).count()
        self._output += "Number of crashes (accidents) in which number of persons killed are male: {res}\n".\
            format(res=result)

    def two_wheelers_booked_for_crashes(self):
        """
        How many two wheelers are booked for crashes
        """
        result = self.units_use_df.filter((self.units_use_df.VEH_BODY_STYL_ID == "MOTORCYCLE") | (
                self.units_use_df.VEH_BODY_STYL_ID == "POLICE MOTORCYCLE")).count()
        self._output += "Number of two wheelers booked for crashes: {res}\n".format(res=result)

    def state_with_highest_accidents_females(self):
        """
        Which state has highest number of accidents in which females are involved
        """
        result = self.primary_person_use_df.filter(self.primary_person_use_df.PRSN_GNDR_ID == "FEMALE"). \
            groupBy("DRVR_LIC_STATE_ID").agg(count("CRASH_ID").alias("CRASH_COUNT")). \
            orderBy("CRASH_COUNT", ascending=False).select("DRVR_LIC_STATE_ID").limit(1).collect()[0]
        self._output += "State that has highest number of accidents in which females are involved: {res}\n".format(
            res=result)

    def top_5_to_15_vehicle_ids_largest_no_of_injuries(self):
        """
        Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        """
        df = self.units_use_df.withColumn("count",
                                          self.units_use_df.TOT_INJRY_CNT + self.units_use_df.DEATH_CNT).orderBy(
            "count", ascending=False).limit(15)
        w = Window().orderBy(col("count").desc())
        result = df.withColumn("row_num", row_number().over(w)).filter(col("row_num").between(5, 15)).select(
            "VEH_MAKE_ID").collect()
        self._output += "Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including " \
                        "death: {res}\n".format(res=result)

    def top_ethnic_user_group_of_each_unique_body_style(self):
        """
        For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
        """
        # unable to find MAX ethnicity group count for each body style
        result = self.units_use_df.join(self.primary_person_use_df,
                               self.units_use_df.CRASH_ID == self.primary_person_use_df.CRASH_ID).groupBy(
            "VEH_BODY_STYL_ID", "PRSN_ETHNICITY_ID").agg(count("PRSN_ETHNICITY_ID").alias("ethn_count")).collect()
        self._output += "For all the body styles involved in crashes, the top ethnic user group of each unique body " \
                        "style: {res}\n".format(res=result)

    def top_5_zip_codes_with_highest_no_of_crashes_with_alcohol_factor(self):
        """
        Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols
        as the contributing factor to a crash (Use Driver Zip Code)
        """
        result = self.primary_person_use_df.join(self.units_use_df, "CRASH_ID", "inner").where(
            ((col("VEH_BODY_STYL_ID") == "PASSENGER CAR, 4-DOOR") | (
                        col("VEH_BODY_STYL_ID") == "PASSENGER CAR, 2-DOOR"))
            & (col("PRSN_ALC_RSLT_ID") == "Positive")).groupBy(
            col("DRVR_ZIP")).agg(count("CRASH_ID").alias("CRASH_COUNT")).orderBy("CRASH_COUNT", ascending=False).\
            select("DRVR_ZIP").limit(5).collect()
        self._output += "Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with " \
                        "alcohols as the contributing factor to a crash: {res}\n".format(res=result)

    def count_distinct_crash_ids_with_damages(self):
        """
        Count of Distinct Crash IDs where No Damaged Property was observed
        and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance
        """
        damage_levels = {'DAMAGED 4', 'DAMAGED 5', 'DAMAGED 6', 'DAMAGE 7 HIGHEST'}
        result = self.units_use_df.join(self.damages_use_df, "CRASH_ID", "left").filter(
            self.damages_use_df.CRASH_ID.isNull &
            (self.units_use_df.VEH_DMAG_SCL_1_ID.isin(damage_levels)) &
            (self.units_use_df.VEH_DMAG_SCL_2_ID.isin(damage_levels)) &
            (self.units_use_df.FIN_RESP_TYPE_ID != "NA")).select("CRASH_ID").distinct().collect()
        self._output += "Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (" \
                        "VEH_DMAG_SCL~) is above 4 and car avails Insurance: {res}\n".format(res=result)

    def determine_top_5_vehicle_makes(self):
        """
        Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences,
        has licensed Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25
        states with highest number of offences (to be deduced from the data)
        """
        invalid_license = {'NA', 'UNLICENSED', 'UNKNOWN'}
        invalid_states = {'NA', 'Unknown'}

        top_10_colors_df = self.units_use_df.groupBy("VEH_COLOR_ID").agg(
            count("CRASH_ID").alias("COLOR_COUNT")).orderBy(
            "COLOR_COUNT", ascending=False).limit(10).select("VEH_COLOR_ID")
        top_10_colors = [row[0] for row in top_10_colors_df.collect()]

        top_25_states_highest_offences_df = self.primary_person_use_df.groupBy("DRVR_LIC_STATE_ID"). \
            agg(count("CRASH_ID").alias("CRASH_COUNT")). \
            orderBy("CRASH_COUNT", ascending=False).filter(~col("DRVR_LIC_STATE_ID").isin(invalid_states)). \
            limit(25).select("DRVR_LIC_STATE_ID")
        top_25_states_highest_offences = [row[0] for row in top_25_states_highest_offences_df.collect()]

        result = self.units_use_df.join(self.charges_use_df, "CRASH_ID", "inner").\
            join(self.primary_person_use_df, "CRASH_ID", "inner").\
            filter(self.charges_use_df.CHARGES.contains("SPEED") &
                   ~self.primary_person_use_df.DRVR_LIC_CLS_ID.isin(invalid_license) &
                   self.primary_person_use_df.DRVR_LIC_STATE_ID.isin(top_25_states_highest_offences) &
                   self.units_use_df.VEH_COLOR_ID.isin(top_10_colors)). \
            groupBy(self.units_use_df.VEH_MAKE_ID).agg(count("CRASH_ID").alias("CRASH_COUNT")). \
            orderBy("CRASH_COUNT", ascending=False).select("VEH_MAKE_ID").limit(5).collect()

        self._output += """Top 5 Vehicle Makes where drivers are charged with speeding related offences,
        has licensed Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25
        states with highest number of offences: {res}\n""".format(res=result)

    def write_to_file(self):
        """
        Method to write the output in the output file path
        """
        try:
            self._remove_file_if_exists(self._output_file_path)
            with open(self._output_file_path, "w") as file_writer:
                file_writer.write(self._output)
            print("Output is successfully saved at the path: {path}".format(path=self._output_file_path))
        except Exception as e:
            print("Error occurred while writing output to file: {msg}".format(msg=str(e)))
            raise

    @staticmethod
    def _remove_file_if_exists(path):
        """
        Removes the file at path if it already exists
        :param path: path of the file
        """
        try:
            if os.path.exists(path):
                print("File already exists at path {path}. Removing the existing file...".format(path=path))
                os.remove(path)
                print("File deleted successfully!".format(path=path))
        except Exception as e:
            print("Error occurred while removing file if it exists: {msg}".format(msg=str(e)))

    def execute_analysis(self):
        """
        Performs various analysis on the provided Car crash data
        and writes output to file path provided in the configuration
        """
        try:
            # Analysis 1
            self.get_no_of_car_crashes_persons_killed_male()
            # Analysis 2
            self.two_wheelers_booked_for_crashes()
            # Analysis 3
            self.state_with_highest_accidents_females()
            # Analysis 4
            self.top_5_to_15_vehicle_ids_largest_no_of_injuries()
            # Analysis 5
            self.top_ethnic_user_group_of_each_unique_body_style()
            # Analysis 6
            self.top_5_zip_codes_with_highest_no_of_crashes_with_alcohol_factor()
            # Analysis 7
            self.count_distinct_crash_ids_with_damages()
            # Analysis 8
            self.determine_top_5_vehicle_makes()
            # Write output to file
            self.write_to_file()

        except AnalysisException as analysis_ex:
            ex_msg = 'AnalysisException Exception: {msg}'.format(msg=analysis_ex.desc)
            print(ex_msg)
            raise
        except IllegalArgumentException as illegal_arg_ex:
            ex_msg = 'IllegalArgumentException Exception: {msg}'.format(msg=illegal_arg_ex.desc)
            print(ex_msg)
            raise
        except Exception as e:
            print(str(e))
            raise


if __name__ == "__main__":
    car_crash_analysis_obj = CarCrashAnalysis()
    car_crash_analysis_obj.execute_analysis()
