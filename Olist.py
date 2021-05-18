import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression


#Link to Linkedin Profile using html
html_name = """
<div style = "text-align:center"> <i>
<a style = "color:grey; font-size:10px" href="https://www.linkedin.com/in/karenbassil/">Karen Bassil</a>
</i> </div>
"""
st.markdown(html_name,  unsafe_allow_html=True)

#Header Title using html
html_title = """
<div style = "background-color:#2C3FCE;padding:15px">
<h1 style = "color:#8df0c3; text-align:center; font-size:50px">Welcome to, Olist</h1>
</div>
"""
st.markdown(html_title, unsafe_allow_html=True)

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
def loading_data():
	#Loading datasets for the Brazilian E-Commerce Customers and Orders
	orders = pd.read_csv('Brazilian E-commerce/olist_orders_dataset.csv')
	order_items = pd.read_csv('Brazilian E-commerce/olist_order_items_dataset.csv')
	customer = pd.read_csv('Brazilian E-commerce/olist_customers_dataset.csv')
	order_payments = pd.read_csv('Brazilian E-commerce/olist_order_payments_dataset.csv')
	products = pd.read_csv('Brazilian E-commerce/olist_products_dataset.csv')
	products_english_name = pd.read_csv('Brazilian E-commerce/product_category_name_translation.csv')
	geolocation_customer = pd.read_csv('Brazilian E-commerce/olist_geolocation_customer_dataset.csv')
	return orders, order_items, customer, order_payments, products, products_english_name, geolocation_customer

orders, order_items, customer, order_payments, products, products_english_name, geolocation_customer = loading_data()

#Converting column type to datetime
time = ['order_purchase_timestamp',
       'order_approved_at', 'order_delivered_carrier_date',
       'order_delivered_customer_date', 'order_estimated_delivery_date']

for t in time:
    orders[t] = pd.to_datetime(orders[t])

#Merging some datasets to have a full data for customers details and items
df1 = orders.merge(order_items, on='order_id', how='inner')
order_item_customer = df1.merge(customer, on='customer_id', how='inner')


st.write("")
st.write("""Olist is an online e-commerce service based in Brazil and it was created in 2015
	with the mission of helping each and every shopkeeper to reach the biggest 
	and best marketplaces nationals and internationals.
	In this dashboard, we will be exploring Olist from January 2017 till August 2018.""")

st.header("Olist's evolution over time")

#Adding year, month, day, day of week columns
order_item_customer['Year'] = order_item_customer['order_purchase_timestamp'].dt.year
order_item_customer['Month'] = order_item_customer['order_purchase_timestamp'].dt.month
order_item_customer['Day'] = order_item_customer['order_purchase_timestamp'].dt.day
order_item_customer['Day of Week'] = order_item_customer['order_purchase_timestamp'].dt.dayofweek
order_item_customer['Hour'] = order_item_customer['order_purchase_timestamp'].dt.hour

#Assigning labels indicating the part of day for each timeframe
hour_bins = [0,5,8,12,18,23]
labels = ['Late Night', 'Early Morning','Morning','After Noon','Night']
order_item_customer['Part of Day'] = pd.cut(order_item_customer['Hour'], bins=hour_bins, labels=labels, include_lowest=True)

#Extracting subset from year 2017-01 till 2018-08
order_item_customer = order_item_customer[(order_item_customer['Year']>2016)]
order_item_customer = order_item_customer[~((order_item_customer['Year']>=2018) & (order_item_customer['Month']>=9))]

#Getting total number of unique customers
unique_customers = len(order_item_customer['customer_unique_id'].unique())
#Getting total number of product quantities solds
total_quantity_sold = len(order_item_customer['order_id'])
#Getting aggregate of total sales/revenue at olis
total_sales = sum(order_item_customer['price'])

col1, col2, col3 = st.beta_columns(3)
col1.markdown("""<h3 style = "color:#79e0b1" </h3>"""+ str(unique_customers), unsafe_allow_html=True)
col1.write('Unique Customers')
col2.markdown("""<h3 style = "color:#79e0b1" </h3>"""+ str(total_quantity_sold), unsafe_allow_html=True)
col2.write('Total Quantities Sold')
col3.markdown("""<h3 style = "color:#79e0b1" </h3>"""+ str(round(total_sales,2)), unsafe_allow_html=True)
col3.write('Total Sales Revenue')

st.write("")

#Getting total number of orders per year and month
order_count_ym = order_item_customer.groupby(['Year', 'Month'], as_index=False)['order_id'].count()
#Joining year and months columns
order_count_ym['YearMonth'] = (order_count_ym['Year'].astype(str)) + '-' + (order_count_ym['Month'].astype(str))
#Getting total number of orders per year, month and days
order_count_ymd = order_item_customer.groupby(['Year', 'Month','Day'], as_index=False)['order_id'].count()
#Joining year, months and days columns
order_count_ymd['YearMonthDays'] = (order_count_ymd['Year'].astype(str)) + '-' + (order_count_ymd['Month'].astype(str)) + '-' + (order_count_ymd['Day'].astype(str))

col1, col2, col3 = st.beta_columns(3)

period = col3.radio('Filter by:', ('Month', 'Day'))

if period == 'Month':
	month_period = 20
	#Getting total number of product quantities solds
	average_quantity_sold = len(order_item_customer['order_id'])/month_period
	#Getting aggregate of total sales/revenue at olis
	average_sales = sum(order_item_customer['price'])/month_period
else:
	day_period = 601
	#Getting total number of product quantities solds
	average_quantity_sold = len(order_item_customer['order_id'])/day_period
	#Getting aggregate of total sales/revenue at olis
	average_sales = sum(order_item_customer['price'])/day_period

col1.markdown("""<h3 style = "color:#2C3FCE" </h3>"""+ str(round(average_quantity_sold,2)), unsafe_allow_html=True)
col1.write('Average Quantities Sold per ' + period)
col2.markdown("""<h3 style = "color:#2C3FCE" </h3>"""+ str(round(average_sales,2)), unsafe_allow_html=True)
col2.write('Average Sales Revenue per ' + period)

st.write("")

if period == 'Month':
	#Line Plot for the total orders across year and month
	data = go.Scatter(mode='lines',
		x = order_count_ym['YearMonth'],
	    y = order_count_ym['order_id'],
	    hovertext = order_count_ym['Month'],
	    text = order_count_ym['Year'],
	    hovertemplate = 'Year: %{text}<br>Month: %{hovertext}<br>Orders: %{y} <extra></extra>')

	#Adding titles, removing gridlines, changing colors,...
	layout = go.Layout(yaxis_title = 'Total Orders',
		autosize = True,
		plot_bgcolor='white',
		xaxis = dict(showgrid=False, linecolor= 'rgb(35,62,139)'),
		yaxis = dict(showgrid=False),
		margin=dict(t=25,l=10,b=1,r=10),
		height = 200, width=700)

	fig = dict(data = data, layout = layout)
	st.plotly_chart(fig)

	st.write("""The Brazilian e-commerce company has been growing rapidly since 2017. 
			Its total orders per month have tremendously increased to even reach 8665 orders in the 
			month of November 2017, and after which purchases have witnessed a steady pace.""")
else:
	#Line Plot for the total orders across year and month and day
	data = go.Scatter(mode='lines',
		x = order_count_ymd['YearMonthDays'],
	    y = order_count_ymd['order_id'],
	    hovertext = order_count_ymd['Month'],
	    text = order_count_ymd['Day'],
	    hovertemplate = 'Month: %{hovertext}<br>Day: %{text}<br>Orders: %{y} <extra></extra>')

	#Adding titles, removing gridlines, changing colors,...
	layout = go.Layout(yaxis_title = 'Total Orders',
		autosize = True,
		plot_bgcolor='white',
		xaxis = dict(showgrid=False, linecolor= 'rgb(35,62,139)'),
		yaxis = dict(showgrid=False),
		margin=dict(t=25,l=10,b=1,r=10),
		height = 200, width=700)

	fig = dict(data = data, layout = layout)
	st.plotly_chart(fig)

	st.write("""The Brazilian e-commerce company has been growing rapidly since 2017. 
			Its total orders per day have tremendously increased to even reach 1366 orders on 24 
			November 2017, and after which purchases have witnessed a steady pace.""")


st.write("")
st.header('Customers Active Time')
#Getting total orders per day of week
order_count_day = order_item_customer.groupby(['Day of Week'], as_index=False)['order_id'].count()
st.write('')
col1, col2 = st.beta_columns((1,1))

#Assigning a list of colors to pass it for the layout
colors = ['rgb(43, 77, 173)',] * 7
#Changing column 0's color as it has the highest total orders
colors[0] = 'rgb(133, 220, 180)'

#Bar Plot for the total orders across year and month
data = go.Bar(x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    y = order_count_day['order_id'],
    text = round((order_count_day['order_id']/sum(order_count_day['order_id']))*100,2),
    texttemplate = '%{text}%',
    textposition='outside',
    hovertemplate = 'Day of Week: %{x}<br>Orders: %{y}<extra></extra>',
    marker_color = colors)

#Adding titles, removing gridlines,...
layout = go.Layout(yaxis_title = 'Total Orders',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False),
	yaxis = dict(showgrid=False, range=(0,19800)),
	margin=dict(t=10,l=10,b=10,r=10),
	height = 260, width =370)

fig = dict(data = data, layout = layout)
col1.plotly_chart(fig)

#Total orders per part of day
order_count_hour = order_item_customer.groupby(['Part of Day'], as_index=False)['order_id'].count()

#Assigning a list of colors
colors = ['rgb(43, 77, 173)',] * 5
#The fourth column presents the highest orders so we will color it with a different color
colors[3] = 'rgb(133, 220, 180)'

#Bar Plot for the total orders across year and month
data = go.Bar(x = order_count_hour['Part of Day'],
    y = order_count_hour['order_id'],
    text = round((order_count_hour['order_id']/sum(order_count_hour['order_id']))*100,2),
    texttemplate = '%{text}%',
    textposition='outside',
    hovertemplate = 'Part of Day: %{x}<br>Orders: %{y}<extra></extra>',
    marker_color = colors)

#Adding titles, removing gridlines, changing colors,...
layout = go.Layout(yaxis_title = 'Total Orders',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False),
	yaxis = dict(showgrid=False, range=(0,47000)),
	margin=dict(t=10,l=10,b=10,r=10),
	height = 265, width =340)

fig = dict(data = data, layout = layout)
col2.plotly_chart(fig)

st.write("""Monday is the day on which Olist's customers tend to purchase the most, whereas Saturday is 
	found to be the least day on which only 10.8% of orders were placed. On the other hand, clients are more 
	prone to shop during the after noon where almost 38.67% of orders have been placed.""")

#Total number of orders per City
order_count_city = order_item_customer.groupby(['customer_city'], as_index=False)['order_id'].count()
order_count_city = order_count_city.rename(columns={'customer_city': 'customer_geolocation_city'})
#Merging with lat and long
order_location = order_count_city.merge(geolocation_customer, on='customer_geolocation_city', how='left')

#Removing duplicates
order_location.drop_duplicates(subset=['customer_geolocation_city', 'order_id'], keep = 'first',
                               inplace= True, ignore_index = True)

#Removing rows where latitude is NaN
order_location = order_location[~order_location['customer_geolocation_lat'].isna()].reset_index()

st.header("Orders Overview by Location")

style = st.selectbox('Select figure type:', ('Map', 'Bar Chart'))
loc = st.selectbox('Select location:', ('States', 'Cities'))

#Sorting dataframe to get top 10 and low 10
top10_cities = order_location.sort_values('order_id', ascending=False)[:10]
low10_cities = order_location.sort_values('order_id')[:10]

#Grouping by state
top10_states = order_location.groupby('customer_geolocation_state', as_index=False)['order_id'].sum().sort_values('order_id', ascending=False)[:10]
low10_states = order_location.groupby('customer_geolocation_state', as_index=False)['order_id'].sum().sort_values('order_id')[:10]

top10_states = top10_states.merge(order_location, on = 'customer_geolocation_state', how = 'left').sort_values(['order_id_y','customer_geolocation_state'], ascending = False).drop_duplicates('customer_geolocation_state', keep='first')
low10_states = low10_states.merge(order_location, on = 'customer_geolocation_state', how = 'left').sort_values(['order_id_y','customer_geolocation_state'], ascending = False).drop_duplicates('customer_geolocation_state', keep='first')

top10_states = top10_states.rename(columns = {'order_id_x': 'order_id'})
low10_states = low10_states.rename(columns = {'order_id_x': 'order_id'})

top10_states = top10_states.sort_values('order_id', ascending = False)
low10_states = low10_states.sort_values('order_id', ascending = False)

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
#Creating a function to plot a Map for the order of the top or lowest countries
def order_map(df, order, locate):

	data = []

	if locate == 'States':
		location_list = df['customer_geolocation_state']
	elif locate == 'Cities':
		location_list = df['customer_geolocation_city']

	for i, place in enumerate(location_list):
		#Saving latitude and longitutde
		if locate == 'States':
			lat = df[df['customer_geolocation_state']==place]['customer_geolocation_lat']
			lon = df[df['customer_geolocation_state']==place]['customer_geolocation_lng']
			count = df[df['customer_geolocation_state']==place].iloc[0,1]
		elif locate == 'Cities':
			lat = df[df['customer_geolocation_city']==place]['customer_geolocation_lat']
			lon = df[df['customer_geolocation_city']==place]['customer_geolocation_lng']
			count = df[df['customer_geolocation_city']==place].iloc[0,2]

		#Based on top or low we will be assigning the size of the circle on the map
		if order == 'Top 10':
			count_size = (30 -2*i)
		elif order == 'Lowest 10':
			count_size = 15

		#Assigning a distinct color for sao paulo as it is the city with highest total orders
		if (place == 'sao paulo') or (place == 'SP'):
			color = '#2C3FCE'
		else:
			color = '#7dd1ab'

		#Plotting long and lat for every distinct city in Olist Brazil
		data.append(go.Scattermapbox(lat = lat,
			lon = lon, 
 			mode = 'markers',
 			marker = dict(size = count_size, color = color, opacity = 0.8), 
 			text = str(place), 
 			name = str(count),
 			hoverinfo = ('name + text')))

	mapbox_access_token = 'pk.eyJ1Ijoia2FyZW5iYXNzaWwiLCJhIjoiY2tsMmZuYXh6MDdyNzJvcndmOHV1d240cSJ9.r9mG8nrg1xYvls7_jSPEmA'

	layout = go.Layout(title = order +' '+ locate +' with regard to total orders',
		autosize = True, 
		hovermode = 'closest', 
		showlegend = False, 
		margin=dict(t=30,l=10,b=70,r=110),
		mapbox = dict(accesstoken = mapbox_access_token, 
			bearing = 0,
			center = dict(lat = -23.55, lon = -46.63), 
			pitch = 0, zoom = 4, style = 'light'))

	fig = dict(data = data, layout = layout)

	return fig

#Creating a function to plot a bar chart for the order of the top or lowest countries
def order_bar(df, order, locate):

	if locate == 'States':
		location_list = df['customer_geolocation_state']
		if order == 'Top 10':
			colors = ['rgb(43, 77, 173)',] * 10
			#The first bar presents the highest orders so we will color it with a different color
			colors[0] = 'rgb(133, 220, 180)'
			y_range = (0, 51000)
		elif order == 'Lowest 10':
			colors = ['rgb(43, 77, 173)',] * 10
			#The 10th bar presents the lowest orders so we will color it with a different color
			colors[9] = 'rgb(133, 220, 180)'
			y_range = (0,1000)
	elif locate == 'Cities':
		location_list = df['customer_geolocation_city']
		if order == 'Top 10':
			colors = ['rgb(43, 77, 173)',] * 10
			#The first bar presents the highest orders so we will color it with a different color
			colors[0] = 'rgb(133, 220, 180)'
			y_range = (0, 19000)
		elif order == 'Lowest 10':
			colors = ['rgb(43, 77, 173)',] * 10
			y_range = (0,6)

	#Bar Plot for the total orders across year and month
	data = go.Bar(x = location_list,
		y = df['order_id'],
    	text = round((df['order_id']/len(order_item_customer['order_id']))*100,2),
    	texttemplate = '%{text}%',
    	textposition='outside',
    	hovertemplate = 'City: %{x}<br>Orders: %{y}<extra></extra>',
    	marker_color = colors)

	#Adding titles, removing gridlines, changing colors,...
	layout = go.Layout(title = order +' '+ locate + ' with regards to total orders',
		yaxis_title = 'Total Orders',
		autosize = True,
		plot_bgcolor='white',
		xaxis = dict(showgrid=False),
		yaxis = dict(showgrid=False, range=y_range),
		margin=dict(t=50,l=20,b=10,r=10),
		height = 330, width =550)

	fig = dict(data = data, layout = layout)

	return fig

col1, col2 = st.beta_columns((1,4))

col1.write("")
col1.write("")

button = col1.radio('Which cities would you like to visualize?', ('Top 10', 'Lowest 10'))

if style == 'Map':
	if loc == 'States':
		if button == 'Top 10':
			map_fig = order_map(top10_states, order = 'Top 10', locate = 'States')
			text_fig = """The State of São Paulo is identified to be the greatest state in 
			terms of total orders where 42.27% of all purchases where originated from São Paulo. 
			It is worth mentioning that approximately 90% of all orders are from customers living 
			in the top 10 states and almost 68% of orders are from the top 3
			following states: São Paulo, Rio de Janeiro, and Minas Gerais."""
		elif button == 'Lowest 10':
			map_fig = order_map(low10_states, order = 'Lowest 10', locate = 'States')
			text_fig = """The lowest 10 states account for only 3.26% of total placed orders and from 
			which we can detect that the State of Amapá had recorded only 86 transaction out of all purchases."""
	elif loc == 'Cities':
		if button == 'Top 10':
			map_fig = order_map(top10_cities, order = 'Top 10', locate = 'Cities')
			text_fig = """In terms of cities, São Paulo accounts for 15.82% of total orders and for which 
			we can assume that it is the city with the most loyal and active customers on Olist's webiste."""
		elif button == 'Lowest 10':
			map_fig = order_map(low10_cities, order = 'Lowest 10', locate = 'Cities')
			text_fig = """On the other hand, an abundant number of cities is available where only 1 order 
			has been placed in the last year and a hlaf. The latter urges Olist to consider expanding to cities 
			where there is low out-reach and identify prominent regions to invest in."""
elif style == 'Bar Chart':
	if loc == 'States':
		if button == 'Top 10':
			map_fig = order_bar(top10_states, order = 'Top 10', locate = 'States')
			text_fig = """The State of São Paulo is identified to be the greatest state in 
			terms of total orders where 42.27% of all purchases where originated from São Paulo. 
			It is worth mentioning that approximately 90% of all orders are from customers living 
			in the top 10 states and almost 68% of orders are from the top 3
			following states: São Paulo, Rio de Janeiro, and Minas Gerais."""
		elif button == 'Lowest 10':
			map_fig = order_bar(low10_states, order = 'Lowest 10', locate = 'States')
			text_fig = """The lowest 10 states account for only 3.26% of total placed orders and from 
			which we can detect that the State of Amapá had recorded only 86 transaction out of all purchases."""
	elif loc == 'Cities':
		if button == 'Top 10':
			map_fig = order_bar(top10_cities, order = 'Top 10', locate = 'Cities')
			text_fig = """In terms of cities, São Paulo accounts for 15.82% of total orders and for which 
			we can assume that it is the city with the most loyal and active customers on Olist's webiste."""
		elif button == 'Lowest 10':
			map_fig = order_bar(low10_cities, order = 'Lowest 10', locate = 'Cities')
			text_fig = """On the other hand, an abundant number of cities is available where only 1 order 
			has been placed in the last year and a hlaf. The latter urges Olist to consider expanding to cities 
			where there is low out-reach and identify prominent regions to invest in."""

col2.plotly_chart(map_fig)
st.write(text_fig)

st.header('Product Categories Overview')

#Saving product id and name only without product physical information
products = products[['product_id','product_category_name']]
#Merging dataframes to get the products names in english
eng_products = products.merge(products_english_name, on = 'product_category_name')
#Merging dataframes to have products and orders together
product = eng_products.merge(order_items, on = 'product_id')

top_products = product.groupby('product_category_name_english', as_index=False)['order_id'].count().sort_values('order_id', ascending=False)
low_products = product.groupby('product_category_name_english', as_index=False)['order_id'].count().sort_values('order_id')

col1, col2 = st.beta_columns((1,4))
order_option = col1.selectbox('Order Option:', ('Top', 'Low'))
num = col2.slider('Slide me:', min_value = 5, max_value = 20)

#Assigning a list of colors
colors = ['rgb(43, 77, 173)',] * num

if order_option == 'Top':
	data = top_products[0:num].sort_values('order_id')
	#The first bar presents the highest orders so we will color it with a different color
	colors[num-1] = 'rgb(133, 220, 180)'
	#Getting max total orders in the subset data
	mx = data['order_id'].max()
	#Adding range of y_axis
	range_x = (0,mx+1100)
	#Assinging a title for the graph
	title_graph = "Top " + str(num) + " product categories"
	#Assigning r for rounding purposes for precentage orders
	r = 2
	#Adding explanation
	fig_text = "With " + str(len(top_products['product_category_name_english'].unique())) + """ unique products, 
	we can identify that Olist's customers are mainly interested in buying from the Bed, Bath, and Table categories 
	as well as from the Health Beauty section."""
elif order_option == 'Low':
	data = low_products[0:num].sort_values('order_id', ascending = False)
	#The 10th bar presents the lowest orders so we will color it with a different color
	colors[num-1] = 'rgb(133, 220, 180)'
	#Getting max total orders in the subset data
	mx = data['order_id'].max()
	#Adding range of y_axis
	range_x = (0,mx+15)
	#Assinging a title for the graph
	title_graph = "Lowest " + str(num) + " product categories"
	#Assigning r for rounding purposes for precentage orders
	r = 3
	#Adding explanation
	fig_text = """In contrast, the least popular category at Olist is detected to be related to Security and Services 
	items with only two orders recorded. In addition, Children Fashion Clothers, CDs and DVDs musicals, and Cuisine supplies 
	are also not very appealing categories."""

#Bar Plot for the total orders across year and month
data = go.Bar(x = data['order_id'] ,
    y = data['product_category_name_english'],
    text = round((data['order_id']/sum(top_products['order_id']))*100,r),
    texttemplate = '%{text}%',
    textposition='outside',
    hovertemplate = 'Product Name: %{y}<br>Orders: %{x}<extra></extra>',
    marker_color = colors,
    orientation = 'h')

#Adding titles, removing gridlines, changing colors,...
layout = go.Layout(title = title_graph,
	xaxis_title = 'Total Orders',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False, range = range_x),
	yaxis = dict(showgrid=False),
	margin=dict(t=30,l=10,b=10,r=15),
	height = 400, width =730)

fig = dict(data = data, layout = layout)

st.write("")
st.plotly_chart(fig)
st.write(fig_text)

st.header('Price & Freight Cost per Product Category')

to_display = st.multiselect('Select metric(s) to display:', ('Price', 'Freight value'))

if to_display == ['Price', 'Freight value']:
	sort_by = st.selectbox('Sort by: ', ('Price', 'Freight'))

col1, col2, col3 = st.beta_columns((1,1,3))
computation = col1.selectbox('Metric(s) Computation:', ('Sum', 'Average'))
order_option_2 = col2.selectbox('Order Option :', ('Top', 'Low'))
num_2 = col3.slider('Slide me :', min_value = 5, max_value = 20)

if computation == 'Sum':
	price_freight = product.groupby('product_category_name_english', as_index=False)[['price', 'freight_value']].sum()
elif computation == 'Average':
	price_freight = product.groupby('product_category_name_english', as_index=False)[['price', 'freight_value']].mean()

#Assigning a list of colors
colors = ['rgb(43, 77, 173)',] * num_2

#Based on selected filters, a plot will be generated
if to_display == []:
	st.markdown("<i>Please select at least one metric in order to generate a visual </i>" , unsafe_allow_html=True)
else:
	if to_display == ['Price', 'Freight value']:
		datas = []

		if sort_by == 'Price':
			sort_by_col = 'price'
		elif sort_by == 'Freight':
			sort_by_col = 'freight_value'

		if order_option_2 == 'Top':
			data = price_freight.sort_values(sort_by_col,  ascending = False)[0:num_2].sort_values(sort_by_col)
			#Getting max total in the subset data
			mx = data['price'].max()
			#Adding range of y_axis
			if computation == 'Sum':
				range_x = (0,mx+250000)
			elif computation == 'Average':
				range_x = (0,mx+100)
			#Assinging a title for the graph
			title_graph = "Top " + str(num_2) + " product categories"
			#Assigning r for rounding purposes for precentage orders
			r = 2
		elif order_option_2 == 'Low':
			data = price_freight.sort_values(sort_by_col)[0:num_2].sort_values(sort_by_col,  ascending = False)
			#Getting max total in the subset data
			mx = data['price'].max()
			#Adding range of y_axis
			if computation == 'Sum':
				range_x = (0,mx+1100)
			elif computation == 'Average':
				range_x = (0,mx+20)
			#Assinging a title for the graph
			title_graph = "Lowest " + str(num_2) + " product categories"
			#Assigning r for rounding purposes for precentage orders
			r = 3

		#Bar Plot for the total price across products
		datas.append(go.Bar(x = data['price'] ,
			y = data['product_category_name_english'],
	    	name='Price',
	    	hovertemplate = 'Product Name: %{y}<br>'+ computation + ' Price: %{x}<extra></extra>',
	    	marker_color = 'rgb(133, 220, 180)',
	    	orientation = 'h') )

		#Bar Plot for the total freight cost across products
		datas.append(go.Bar(x = data['freight_value'] ,
			y = data['product_category_name_english'],
	    	name='Freight Cost',
	    	hovertemplate = 'Product Name: %{y}<br>'+ computation + ' Freight cost: %{x}<extra></extra>',
	    	marker_color = 'rgb(43, 77, 173)',
	    	orientation = 'h') )

		#Adding titles, removing gridlines, changing colors,...
		layout = go.Layout(title = title_graph,
			xaxis_title = computation + ' Price & Freight Cost',
			autosize = True,
			barmode='stack',
			plot_bgcolor='white',
			xaxis = dict(showgrid=False, range = range_x),
			yaxis = dict(showgrid=False),
			margin=dict(t=30,l=10,b=10,r=15),
			height = 400, width =730)

		fig = dict(data = datas, layout = layout)

		st.write("")
		st.plotly_chart(fig)

	else:

		if to_display[0] == 'Price':
			col = 'price'
		elif to_display[0] == 'Freight value':
			col = 'freight_value'

		if order_option_2 == 'Top':
			data = price_freight[['product_category_name_english', col]].sort_values(col,  ascending = False)[0:num_2].sort_values(col)
			#The first bar presents the highest orders so we will color it with a different color
			colors[num_2-1] = 'rgb(133, 220, 180)'
			#Getting max total orders in the subset data
			mx = data[col].max()
			#Assigning r for rounding purposes for precentage orders
			r = 2
			#Adding range of y_axis
			if computation == 'Sum':
				range_x = (0,mx+110000)
				txt = round((data[col]/sum(price_freight[col]))*100,r)
				hovertext = '%{text}%'
			elif computation == 'Average':
				range_x = (0,mx+100)
				txt = ' '
				hovertext = '%{text}'
			#Assinging a title for the graph
			title_graph = "Top " + str(num_2) + " product categories"
		elif order_option_2 == 'Low':
			data = price_freight[['product_category_name_english', col]].sort_values(col)[0:num_2].sort_values(col,  ascending = False)
			#The 10th bar presents the lowest orders so we will color it with a different color
			colors[num_2-1] = 'rgb(133, 220, 180)'
			#Getting max total orders in the subset data
			mx = data[col].max()
			#Adding range of y_axis
			if computation == 'Sum':
				range_x = (0,mx+800)
				r = 3
				txt = round((data[col]/sum(price_freight[col]))*100,r)
				hovertext = '%{text}%'
			elif computation == 'Average':
				range_x = (0,mx+20)
				txt = ' '
				hovertext = '%{text}'
			#Assinging a title for the graph
			title_graph = "Lowest " + str(num_2) + " product categories"
			#Assigning r for rounding purposes for precentage orders

		#Bar Plot for the total orders across year and month
		data = go.Bar(x = data[col] ,
	    	y = data['product_category_name_english'],
	    	text = txt,
	    	texttemplate = hovertext,
	    	textposition='outside',
	    	hovertemplate = 'Product Name: %{y}<br>' + computation + " " + to_display[0] + ': %{x}<extra></extra>',
	    	marker_color = colors,
	    	orientation = 'h')

		#Adding titles, removing gridlines, changing colors,...
		layout = go.Layout(title = title_graph,
			xaxis_title = computation + " " + to_display[0],
			autosize = True,
			plot_bgcolor='white',
			xaxis = dict(showgrid=False, range = range_x),
			yaxis = dict(showgrid=False),
			margin=dict(t=30,l=10,b=10,r=15),
			height = 400, width =730)

		fig = dict(data = data, layout = layout)

		st.write("")
		st.plotly_chart(fig)

st.header('Orders Shipment and Payment')

shipment = orders.groupby('order_status', as_index=False)['order_id'].count()
shipment = shipment[shipment['order_status'] != 'unavailable']
#Capitilizing first letter
shipment['order_status'] = shipment['order_status'].str.capitalize()
payment = order_payments.groupby('payment_type', as_index=False)['order_id'].count()
payment = payment[payment['payment_type'] != 'not_defined']
#Remocing _ and replace it by space and then capitilize first letter
payment['payment_type'] = payment['payment_type'].replace('_', ' ', regex=True).str.capitalize()

col1, col2 = st.beta_columns((1.5, 1))
#Assigning a list of colors
colors_ship = ['rgb(43, 77, 173)',] * 7
colors_pay = ['rgb(43, 77, 173)',] * 4
#The column x presents the highest count so we will color it with a different color
colors_ship[3] = 'rgb(133, 220, 180)'
colors_pay[1] = 'rgb(133, 220, 180)'

#Bar Plot for the total orders across orders status
data_ship = go.Bar(x = shipment['order_status'],
    y = shipment['order_id'],
    text = round((shipment['order_id']/sum(shipment['order_id']))*100,2),
    texttemplate = '%{text}%',
    textposition='outside',
    hovertemplate = 'Order Status: %{x}<br>Orders: %{y}<extra></extra>',
    marker_color = colors_ship)

#Adding titles, removing gridlines, changing colors,...
layout_ship = go.Layout(yaxis_title = 'Total Orders',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False),
	yaxis = dict(showgrid=False, range=(0,110000)),
	margin=dict(t=10,l=10,b=10,r=10),
	height = 265, width =420)

fig_ship = dict(data = data_ship, layout = layout_ship)
col1.plotly_chart(fig_ship)
col1.write("""Investigating the purchases made between January 2017 and August 2018, we can identify 
	that the bulk of orders were appropriately delivered. The latter suggests that Olist presents a proper and efficient 
	logistics strategy that allow all purchases to be delivered without any problem.""")

#Bar Plot for the total orders across payment type
data_pay = go.Bar(x = payment['payment_type'],
    y = payment['order_id'],
    text = round((payment['order_id']/sum(payment['order_id']))*100,2),
    texttemplate = '%{text}%',
    textposition='outside',
    hovertemplate = 'Order Status: %{x}<br>Orders: %{y}<extra></extra>',
    marker_color = colors_pay)

#Adding titles, removing gridlines, changing colors,...
layout_pay = go.Layout(yaxis_title = 'Total Orders',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False),
	yaxis = dict(showgrid=False, range=(0,85000)),
	margin=dict(t=10,l=10,b=10,r=10),
	height = 265, width =300)

fig_pay = dict(data = data_pay, layout = layout_pay)
col2.plotly_chart(fig_pay)
col2.write("""The Majority of Olist's customers tend to pay using a Credit card and some using a Boleta.""")



st.header('Time to deliver')

time = orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date']]
time = time.copy()
time['delivery_days'] = (time['order_delivered_customer_date'] - time['order_purchase_timestamp']).dt.days
time = time[~time['delivery_days'].isna()]

#Histogram Plot for the time to deliver
data = go.Histogram(x = time['delivery_days'],
    marker_color = 'rgb(43, 77, 173)')

#Adding titles, removing gridlines, changing colors,...
layout = go.Layout(yaxis_title = 'Total Orders',
	xaxis_title = 'Days to Deliver',
	autosize = True,
	plot_bgcolor='white',
	xaxis = dict(showgrid=False),
	yaxis = dict(showgrid=False),
	margin=dict(t=10,l=10,b=10,r=10),
	height = 265, width =700)

fig = dict(data = data, layout = layout)
st.plotly_chart(fig)
st.write("""As identified before, the majority of orders were delivered properly to clients, 
	however, were they delivered on time? Here we can identify that the time to deliver data is rightly-skewed. 
	In other words, on average, it would take 7 days for the shipment to arrive, but some even take so much longer 
	with 209 days being the longest time to deliver. For further investigation, you can inspect below the data 
	by giving a threshold filter for the days to deliver.""")

col1, col2 = st.beta_columns((2,1))
col1.write('')
col1.write('')
check_box = col1.checkbox('Show and insepct data for days to deliver')
num_days = col2.number_input('Enter a number to filter days to deliver')


if check_box:
	#Merging several data to get information
	delay = time.merge(order_items, on = 'order_id').merge(eng_products, on = 'product_id')
	delay = delay.merge(customer, on = 'customer_id')	
	#Dropping uneccessary columns
	delay = delay.drop(['customer_id','product_id','seller_id', 'shipping_limit_date', 'order_item_id', 'product_category_name', 'customer_unique_id', 'customer_zip_code_prefix'], axis=1)	
	#Changing column names
	delay.columns = ['Order ID', 'Order Status', 'Purchase Date', 'Delivery Date', 'Days to Deliver', 'Order Price', 'Freight Value', 'Product Name', 'Customer City', 'Customer State']
	#Adding column for price ranges
	price_bins = [0,10,50,200,500,1000,3000, 7000]
	price_labels = ['(0,10[', '(10,50[','(50,200[','(200,500[','(500,1000[','(1000,3000[','(3000,7000[']
	delay['Price Range'] = pd.cut(delay['Order Price'], bins=price_bins, labels=price_labels, include_lowest=True)
	#Converting column category to object
	delay['Price Range'] = delay['Price Range'].astype('object')
	#Adding column for freight value ranges
	freight_bins = [0,10,50,200,500]
	freight_labels = ['(0,10[', '(10,50[','(50,200[','(200,500[']
	delay['Freight Range'] = pd.cut(delay['Freight Value'], bins=freight_bins, labels=freight_labels, include_lowest=True)
	#Converting column category to object
	delay['Freight Range'] = delay['Freight Range'].astype('object')
	#Filtering to get only order with delivery dates greater than 50
	delay_50 = delay[delay['Days to Deliver']>num_days]

	st.write("Group by:")

	#Allowing for the user to group by several options
	col1, col2, col3, col4, col5 = st.beta_columns((1,1.5,1.5,1,1))
	button1 = col1.button('Price')
	button2 = col2.button('Freight value')
	button3 = col3.button('Product Name')
	button4 = col4.button('City')
	button5 = col5.button('State')

	if button1:
		st.dataframe(delay_50.groupby(['Price Range'], as_index=False)['Order ID'].count().rename(columns={'Order ID': 'Count'}))
	elif button2:
		st.dataframe(delay_50.groupby(['Freight Range'], as_index=False)['Order ID'].count().rename(columns={'Order ID': 'Count'}))
	elif button3:
		st.dataframe(delay_50.groupby(['Product Name'], as_index=False)['Order ID'].count().rename(columns={'Order ID': 'Count'}).sort_values('Count', ascending=False))
	elif button4:
		st.dataframe(delay_50.groupby(['Customer City'], as_index=False)['Order ID'].count().rename(columns={'Order ID': 'Count'}).sort_values('Count', ascending=False))
	elif button5:
		st.dataframe(delay_50.groupby(['Customer State'], as_index=False)['Order ID'].count().rename(columns={'Order ID': 'Count'}).sort_values('Count', ascending=False))


st.header('Customer Segmentation')

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
def clustering():
	#Preparing the data by merging sevral datasets and extractin the needed columns to calculate the Customer Lifetime Value
	clv = orders.merge(customer, on='customer_id').merge(order_items, on='order_id')
	clv = clv[['order_id', 'order_purchase_timestamp', 'customer_unique_id', 'price', 'freight_value']]
	clv['total_price'] = clv['price'] + clv['freight_value']
	clv = clv.drop(['price', 'freight_value'], axis=1)

	#get date only without time
	clv['order_purchase_timestamp'] = pd.to_datetime(clv['order_purchase_timestamp'].dt.date)

	#Let's calculate the recency 
	recency = clv.groupby('customer_unique_id', as_index = False)['order_purchase_timestamp'].max()
	recency['Recency'] = (recency['order_purchase_timestamp'].max() - recency['order_purchase_timestamp']).dt.days
	recency.drop('order_purchase_timestamp', axis=1, inplace=True)

	#Initiate model
	km_rc = KMeans(n_clusters = 4, random_state=42)
	#Fit the model
	km_rc.fit(recency[['Recency']])

	#Predict clusters for available customers
	recency['Recency Cluster'] = km_rc.predict(recency[['Recency']])

	#Adding clusters mean recency
	recency_cluster = recency.groupby('Recency Cluster', as_index=False)['Recency'].mean()
	recency_cluster = recency_cluster.rename(columns={'Recency': 'Recency_mean'})
	recency_cluster['Recency_mean'] = round(recency_cluster['Recency_mean'],2)

	#Adding cluster std recency
	rec_std = recency.groupby('Recency Cluster', as_index=False)['Recency'].std()['Recency']
	recency_cluster['Recency_std'] = round(rec_std,2)


	#Let's calculate the Frequency
	frequency = clv.groupby('customer_unique_id', as_index = False)['order_id'].count()
	frequency = frequency.rename(columns={'order_id': 'Frequency'})

	#Initiate model
	km_freq = KMeans(n_clusters = 4, random_state=42)
	#Fit the model
	km_freq.fit(frequency[['Frequency']])

	#Predict clusters for available customers
	frequency['Frequency Cluster'] = km_freq.predict(frequency[['Frequency']])

	#Calculating mean of each frequency cluster
	frequency_cluster = frequency.groupby('Frequency Cluster', as_index=False)['Frequency'].mean()
	frequency_cluster = frequency_cluster.rename(columns={'Frequency': 'Frequency_mean'})
	frequency_cluster['Frequency_mean'] = round(frequency_cluster['Frequency_mean'],2)
	#Calculating standrad deviation of each frequency cluster
	freq_std = frequency.groupby('Frequency Cluster', as_index=False)['Frequency'].std()['Frequency']
	frequency_cluster['Frequency_std'] = round(freq_std,2)


	#Let's calculate the Monetary value
	monetary = clv.groupby('customer_unique_id', as_index = False)['total_price'].sum()
	monetary = monetary.rename(columns={'total_price': 'Monetary Value'})

	#Initiate model
	km_mon = KMeans(n_clusters = 4, random_state=42)
	#Fit the model
	km_mon.fit(monetary[['Monetary Value']])

	#Predict clusters for available customers
	monetary['Monetary Cluster'] = km_mon.predict(monetary[['Monetary Value']])

	#Calculating mean of each monetary cluster
	monetary_cluster = monetary.groupby('Monetary Cluster', as_index=False)['Monetary Value'].mean()
	monetary_cluster = monetary_cluster.rename(columns={'Monetary Value': 'Monetary_mean'})
	monetary_cluster['Monetary_mean'] = round(monetary_cluster['Monetary_mean'],2)
	#Calculating standrad deviation of each monetary cluster
	mon_std = monetary.groupby('Monetary Cluster', as_index=False)['Monetary Value'].std()['Monetary Value']
	monetary_cluster['Monetary_std'] = round(mon_std,2)


	clv_recency = recency.merge(recency_cluster, on='Recency Cluster')
	clv_frequency= frequency.merge(frequency_cluster, on='Frequency Cluster')
	clv_monetary = monetary.merge(monetary_cluster, on='Monetary Cluster')

	#Final data
	RFM = clv_recency.merge(clv_frequency, on='customer_unique_id').merge(clv_monetary,on='customer_unique_id')

	#Calculating average order value
	RFM['Average order value'] = RFM['Monetary Value']/RFM['Frequency']

	#Purchase frequency
	purchase_frequency = sum(RFM['Frequency'])/RFM.shape[0]

	#Retention and Churn rate
	retention = (RFM[RFM['Frequency']>1].shape[0])/(RFM.shape[0])
	churn = 1 - retention

	#Assigning a value for profit margin
	profit_margin = 0.05

	#LTV
	RFM['LTV'] = (RFM['Average order value']*purchase_frequency)/churn
	#CLV
	RFM['CLV'] = RFM['LTV']*profit_margin

	#Initiate model
	km_clv = KMeans(n_clusters = 4, random_state=42)
	#Fit the model
	km_clv.fit(RFM[['CLV']])
	#Predict clusters for LTV
	RFM['CLV Cluster'] = km_clv.predict(RFM[['CLV']])

	#Calculating mean of each LTV cluster
	CLV_cluster = RFM.groupby('CLV Cluster', as_index=False)['CLV'].mean()
	CLV_cluster = CLV_cluster.rename(columns={'CLV': 'CLV_mean'})
	CLV_cluster['CLV_mean'] = round(CLV_cluster['CLV_mean'],2)
	#Calculating standrad deviation of each LTV cluster
	clv_std = RFM.groupby('CLV Cluster', as_index=False)['CLV'].std()['CLV']
	CLV_cluster['CLV_std'] = round(clv_std,2)

	return RFM, km_rc, km_freq, km_mon, km_clv

RFM, km_rc, km_freq, km_mon, km_clv = clustering()

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
def train_model():
	#initialize the model
	lr = LinearRegression()
	#Getting the features and the target
	RFM_x = RFM[['Recency', 'Frequency', 'Monetary Value']]
	RFM_y = RFM['CLV']
	#fit and train the model
	model = lr.fit(RFM_x, RFM_y)

	return model

model = train_model()

#Generating a single number cluster to identify a total of 60 segments
RFM['Segment'] = [''.join([str(x),str(y),str(z)]) for x,y,z in zip(RFM['Recency Cluster'],RFM['Frequency Cluster'],RFM['Monetary Cluster'])]

st.write("""Customers segmentation can be a very powerfull decision making tool. Thus, we have decided to opt for the RFM analysis method, which stands for 
	Recency, Frequency, and Monetary value. These three can be analyzed in order to predict an additional indicator which is the Customer Lifetime Value (CLV). 
	The CLV allows the company to estimate the total worth of an individual customer or a group of customers.""")


text = ("""	Based on our analysis, four clusters were identified regarding the Customer Lifetime Value of Olist's clients. The first cluster include customers with 
CLV less than 12, the second cluster contains customers with CLV between 12 and 36, while the third cluster presentens clients with CLV between 36 and  98. And finally, 
the fourth cluster include customers with CLV greater than 98.""")

#html text
html_text = """
<div style = "background-color:#a6edcd;padding:8px">
<p style = "color:#000000; font-size:15px">""" + text + """</p>
</div>
"""
st.markdown(html_text, unsafe_allow_html=True)

st.write("")
expander_1 = st.beta_expander("Action to take based on CLV prediction")
expander_1.write("Recency: date of last purchase in days")

st.write("")
expander_2 = st.beta_expander("Information regarding Recency, Frequency, and Monetary Value")
expander_2.write("Recency: date of last purchase in days")
expander_2.write("Frequency: total number of orders")
expander_2.write("Monetary Value: total orders value")

st.write("")
st.subheader('Try it yourself:')
col1, col2, col3 = st.beta_columns(3)
rec = col1.number_input("Enter Recency Value")
freq = col2.number_input("Enter Frequency Value")
mon = col3.number_input("Enter Monetary Value")

st.write("")
col1, col2, col3 = st.beta_columns((1,2,1))
#Converting inserted data into dataframe
X_data = pd.DataFrame({'Recency': [rec], 'Frequency' : [freq], ' Monetary Value': [mon]})
if col2.button('Predict Customer Lifetime Value'):
	prediction = model.predict(X_data)
	st.write(prediction[0])



