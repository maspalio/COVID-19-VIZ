#!/usr/bin/env python3

from collections import OrderedDict
import re

import numpy

import pandas
from pandas import DataFrame

import plotly.graph_objects as go
from plotly.graph_objects import Bar, Figure, Scatter
from plotly.subplots import make_subplots


class Covid():
    #
    # Constants.
    #

    CSSE_URL     = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    CATEGORIES   = ["confirmed", "deaths", "recovered"]
    INDEXES      = ["Country/Region", "Province/State", "Date"]
    INDEXES_LHS  = ["Country/Region", "Province/State"]  # FIXME vs INDEXES[0:1]?
    INDEXES_MAIN = ["Country/Region", "Date"]


    #
    # Init.
    #

    def __init__(self, color: dict={}) -> None:
        self.color                = color

        self.raw_df               = {}
        self.molten_df            = {}
        self.per_country_df       = {}
        self.daily_per_country_df = {}
        self.main_df              = {}

        self._build()

    #
    # Build.
    #

    def _build(self) -> None:
        for category in self.CATEGORIES:
            capitalized = category.capitalize()

            self.raw_df[category]               = pandas.read_csv("{}/time_series_19-covid-{}.csv".format(self.CSSE_URL, capitalized))
            self.molten_df[category]            = self._molten_df(self.raw_df[category])
            self.per_country_df[category]       = self._per_country_df(self.molten_df[category], "Total {}".format(capitalized))
            self.daily_per_country_df[category] = self._daily_per_country_df(self.per_country_df[category], "Total {}".format(capitalized), "Daily {}".format(capitalized))

        self._build_main()

    #
    # Builders.
    #

    def _build_main(self) -> None:
        #
        # Merge.
        #

        self.main_df = pandas.merge(self.per_country_df["confirmed"], self.daily_per_country_df["confirmed"], how="left", left_index=True, right_index=True)
        self.main_df = pandas.merge(self.main_df,                     self.daily_per_country_df["deaths"],    how="left", left_index=True, right_index=True)
        self.main_df = pandas.merge(self.main_df,                     self.daily_per_country_df["recovered"], how="left", left_index=True, right_index=True)
        self.main_df = pandas.merge(self.main_df,                     self.per_country_df["deaths"],          how="left", left_index=True, right_index=True)
        self.main_df = pandas.merge(self.main_df,                     self.per_country_df["recovered"],       how="left", left_index=True, right_index=True)

        #
        # Reckon.
        #

        self.main_df["Total Active"] = self.main_df["Total Confirmed"] - self.main_df["Total Deaths"] - self.main_df["Total Recovered"]
        self.main_df["Death Ratio"]  = numpy.round(self.main_df["Total Deaths"] / self.main_df["Total Confirmed"], 2)

        #
        # N.W.O.
        #

        columns = ["Total Active"] + ["Total {}".format(c.capitalize()) for c in self.CATEGORIES] + ["Daily {}".format(c.capitalize()) for c in self.CATEGORIES] + ["Death Ratio"]

        self.main_df = self.main_df[columns]

    def _molten_df(self, df: DataFrame) -> DataFrame:
        """
            Melt incoming DF into [["Country/Region", "Province/State", "Date"] X ["Cases"]] DF.
            
            RHS MM/DD/YYYY columns melted into "Cases" column.
            
            "Date" entries as DateTime instances.
            
            Sorted ascendingly on ["Country/Region", "Province/State"] columns.
            Sorted descendingly on ["Date"] column.
        """

        neo_df = df.drop(["Lat", "Long"], axis="columns")

        neo_df = neo_df.melt(
            id_vars=self.INDEXES_LHS,
            value_vars=None,  # Neither ID'ed nor dropped
            var_name="Date",
            value_name="Cases",
        )

        neo_df["Date"] = pandas.to_datetime(neo_df["Date"])  # Parse

        neo_df.set_index(self.INDEXES, inplace=True)
        neo_df.sort_values(by=self.INDEXES, ascending=[True, True, False], inplace=True)

        return neo_df

    def _per_country_df(self, df: DataFrame, column: str) -> DataFrame:
        """ Per-country DF from molten DF """

        indexes = self.INDEXES_MAIN

        neo_df = df.groupby(indexes)["Cases"].sum().reset_index()
        neo_df.set_index(indexes, inplace=True)

        neo_df.index = neo_df.index.set_levels([neo_df.index.levels[0], neo_df.index.levels[1]])

        neo_df.sort_values(indexes, ascending=[True, False], inplace=True)

        neo_df = neo_df.rename(columns={"Cases":column})

        return neo_df

    def _daily_per_country_df(self, df: DataFrame, df_column: str, neo_name: str) -> DataFrame:
        """ Daily per-country DF from per-country DF """

        neo_df = df.groupby(level=0).diff().fillna(0)
        neo_df = neo_df.rename(columns={df_column:neo_name})

        neo_df[neo_name] *= -1  # due to descending sort for display purposes
        neo_df[neo_name] = pandas.to_numeric(neo_df[neo_name], downcast="integer")

        return neo_df

    #
    # Accessors.
    #

    def columns(self) -> list:
        """ List of columns of main DF """

        return list(self.main_df)

    def countries(self) -> list:
        """ List of countries from main DF """

        return self.main_df.index.unique("Country/Region")

    #
    # Mungers.
    #

    def per_country_max_df(self, column: str) -> DataFrame:
        """ Max per-country DF """

        neo_df = self.main_df.max(level=0)[column].reset_index().set_index("Country/Region")
        neo_df.sort_values(by=column, ascending=False, inplace=True)

        return neo_df

    #
    # Figures.
    #

    def countries_scatters(self, countries: list, column: str, mode: str=None, showlegend: bool=True) -> Figure:
        fig = Figure()

        ratio_re = re.compile(r"(?x: \b Ratio \b )")
        if ratio_re.search(column) is not None:
            fig.update_layout(yaxis=dict(tickformat="%.format.%3f"))
        
        for country in countries:
            country_df = self.main_df.loc[[country]].reset_index(level=0, drop=True)
            
            fig.add_trace(
                Scatter(
                    x=country_df.index,
                    y=country_df[column],
                    name=country,
                    mode=mode,
                    line_color=self.color.get(country, "black"),
                    opacity=0.9,
                    showlegend=showlegend,
                )
            )

        fig.update_layout(title_text=column)

        return fig

    def countries_subplots(self, countries: list) -> Figure:
        subplot = OrderedDict({
            "Total Confirmed":  {"row": 1, "col": 1},
            "Total Active":     {"row": 1, "col": 2},
            "Total Deaths":     {"row": 2, "col": 1},
            "Total Recovered": {"row": 2, "col": 2},
        })

        fig = make_subplots(rows=2, cols=2, subplot_titles=list(subplot))

        for column, column_d in subplot.items():
            row, col = column_d["row"], column_d["col"]

            scatters_fig = self.countries_scatters(
                countries=countries,
                column=column,
                showlegend=(row == 1 and col == 1),
            )

            for data in scatters_fig["data"]:
                fig.append_trace(data, row, col)
        
        return fig

    def countries_bar_chart(self, column: str, count: int=None, exclusions: list=None) -> Figure:
        """ Top N countries bar chart """

        max_df = self.per_country_max_df(column)
        if exclusions is not None:
            max_df = max_df[~max_df.index.isin(exclusions)]
        if count is not None:
            max_df = max_df.head(count)

        fig = Figure(Bar(
            x=max_df.index,
            y=max_df[column],
            text=max_df[column],
            textposition="outside",
        ))

        title = "Top {} Countries by \"{}\" Column".format(count, column)
        if exclusions is not None:
            title = "{} Excluding {}".format(title, exclusions)

        fig.update_layout(title_text=title)

        return fig
