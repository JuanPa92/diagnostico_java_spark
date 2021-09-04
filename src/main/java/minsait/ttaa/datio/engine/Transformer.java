package minsait.ttaa.datio.engine;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.jetbrains.annotations.NotNull;

import static minsait.ttaa.datio.common.Common.*;
import static minsait.ttaa.datio.common.naming.PlayerInput.*;
import static minsait.ttaa.datio.common.naming.PlayerOutput.*;
import static org.apache.spark.sql.functions.*;

public class Transformer extends Writer {
    private SparkSession spark;

    public Transformer(@NotNull SparkSession spark) {
        this.spark = spark;
        Dataset<Row> df = readInput();

        df.printSchema();

        df = rankByNationalityWindow(df);
        df = ageRangeFilter(df);
        df = getPotentialVsOverall(df);
        df = cleanData(df);
        df = columnSelection(df);

        // for show 100 records after your transformations and show the Dataset schema
        df.show(200, false);
        df.printSchema();

        // Uncomment when you want write your final output
        write(df);
    }

    private Dataset<Row> columnSelection(Dataset<Row> df) {
        return df.select(
                shortName.column(),
                longName.column(),
                age.column(),
                heightCm.column(),
                weightKg.column(),
                nationality.column(),
                clubName.column(),
                overall.column(),
                potential.column(),
                teamPosition.column(),
                ageRange.column(),
                rankByNationalityPosition.column(),
                potentialVsOverall.column()
        );
    }

    /**
     * @return a Dataset readed from csv file
     */
    private Dataset<Row> readInput() {
        Dataset<Row> df = spark.read()
                .option(HEADER, true)
                .option(INFER_SCHEMA, true)
                .csv(INPUT_PATH);
        return df;
    }

    /**
     * @param df
     * @return a Dataset with filter transformation applied
     * column rank_by_nationality_position != null && column age_range != null && column potential_vs_overall != null
     */
    private Dataset<Row> cleanData(Dataset<Row> df) {

        Column notNull = rankByNationalityPosition.column().isNotNull().and(
                ageRange.column().isNotNull()
        ).and(
                potentialVsOverall.column().isNotNull()
        );

        Column firstCondition = rankByNationalityPosition.column().lt(3);
        Column secondCondition = potentialVsOverall.column().gt(1.15).and(
            ageRange.column().equalTo("B").or(
            ageRange.column().equalTo("C")
          )
        );
        Column thirdCondition = ageRange.column().equalTo("A").and(
            potentialVsOverall.column().gt(1.25)
        );
        Column fourthCondition = ageRange.column().equalTo("D").and(
            rankByNationalityPosition.column().lt(5)
        );

        df = df.filter(
                notNull.and(
                    firstCondition.or(
                      secondCondition
                    ).or(
                      thirdCondition
                    ).or(
                      fourthCondition
                    )
                )
        );

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have nationality, team_position and overall)
     * @return add to the Dataset the column "rank_by_nationality_position"
     * ranking by nationality and team position and sorting it out by the
     * overall column
     */
    private Dataset<Row> rankByNationalityWindow(Dataset<Row> df) {
        WindowSpec w = Window
                .partitionBy(nationality.column(), teamPosition.column())
                .orderBy(overall.column().desc());

        Column rule = row_number().over(w);

        df = df.withColumn(rankByNationalityPosition.getName(), rule);

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have age column)
     * @return add to the Dataset the column "cat_age_range"
     * by each position value
     * cat A for if age is less than 23
     * cat B for if age is between 23 and 26
     * cat C for if age is between 27 and 31
     * cat D for the rest
     */
    private Dataset<Row> ageRangeFilter(Dataset<Row> df) {

        Column rule = when(age.column().lt(23), "A")
          .when(age.column().lt(27), "B")
          .when(age.column().lt(32), "C")
          .otherwise("D");

        df = df.withColumn(ageRange.getName(), rule);

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have potential and overall column)
     * @return add to the Dataset the column "potential_vs_overall"
     * with the calculation potential/overall
     */
    private Dataset<Row> getPotentialVsOverall(Dataset<Row> df) {

        Column rule = potential.column().$div( overall.column() );

        df = df.withColumn(potentialVsOverall.getName(), rule);

        return df;
    }

}
