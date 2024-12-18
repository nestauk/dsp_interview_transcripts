from typing import List
from typing import Optional
from typing import Union

import altair as alt
import pandas as pd


opacity_condition = alt.condition(alt.datum.Topic == -1, alt.value(0.1), alt.value(0.6))


def create_scatterplot(
    data_viz: pd.DataFrame,
    color: str = "Name:N",
    tooltip: List[str] = ["Name:N", "Description:N", "question:N", "text_clean:N"],
    domain: Optional[List[Union[str, int]]] = None,
    range_: Optional[List[str]] = None,
) -> alt.Chart:
    """
    Display texts as a scatterplot.

    domain and range_ are used for specifying a 3-way colour scale when the
    texts should be coloured by sentiment.

    Args:
        data_viz (pd.DataFrame): The DataFrame containing data to visualize. Must include columns for x and y coordinates,
            along with fields specified in `color` and `tooltip`.
        color (str, optional): Encoding specification for the color channel. Defaults to "Name:N".
        tooltip (List[str], optional): List of fields to display as tooltips. Defaults to ["Name:N", "Description:N", "question:N", "text_clean:N"].
        domain (Optional[List[Union[str, int]]], optional): Custom domain values for the color scale, defining specific categories.
            Defaults to None.
        range_ (Optional[List[str]], optional): Custom color range for the color scale, corresponding to the domain values.
            Defaults to None.

    Returns:
        alt.Chart: An Altair chart object representing the scatterplot, with specified color, tooltips, and interactivity.
    """

    if domain is not None and range_ is not None:
        color = alt.Color(color, scale=alt.Scale(domain=domain, range=range_))
    else:
        color = alt.Color(color)

    fig = (
        alt.Chart(data_viz)
        .mark_circle(size=50)
        .encode(
            x=alt.X(
                "x:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            y=alt.Y(
                "y:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            color=color,
            opacity=opacity_condition,  # Ensure opacity_condition is defined elsewhere
            tooltip=tooltip,
        )
        .properties(width=900, height=600)
        .interactive()
    )

    return fig
