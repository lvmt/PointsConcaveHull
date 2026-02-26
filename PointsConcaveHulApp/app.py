import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_closing, disk
from skimage.measure import find_contours
from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import unary_union, polygonize
import base64
import io

# 初始化 Dash 应用，使用 Bootstrap 主题
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 禁用默认的 suppress_callback_exceptions
app.config.suppress_callback_exceptions = True

# 全局变量存储数据
current_data = None

# 凹包计算函数（鲁棒版）
def alpha_shape_algo(points, alpha):
    """
    Robust alpha shape using Delaunay + circumradius filter + boundary polygonize.
    返回轮廓坐标 (N, 2)，不闭合（不重复首尾点）。
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (N,2) array")

    if len(points) < 4 or alpha is None or alpha <= 0:
        hull = MultiPoint(points).convex_hull
        return np.asarray(hull.exterior.coords)[:-1]

    def circumradius(tri_pts):
        a, b, c = tri_pts
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        s = (ab + bc + ca) / 2.0
        area2 = s * (s - ab) * (s - bc) * (s - ca)
        if area2 <= 1e-20:
            return np.inf
        area = np.sqrt(area2)
        return (ab * bc * ca) / (4.0 * area)

    tri = Delaunay(points)

    edge_count = {}

    def add_edge(i, j):
        if i > j:
            i, j = j, i
        edge_count[(i, j)] = edge_count.get((i, j), 0) + 1

    thresh = 1.0 / alpha
    for simplex in tri.simplices:
        tri_pts = points[simplex]
        if circumradius(tri_pts) < thresh:
            add_edge(simplex[0], simplex[1])
            add_edge(simplex[1], simplex[2])
            add_edge(simplex[2], simplex[0])

    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        hull = MultiPoint(points).convex_hull
        return np.asarray(hull.exterior.coords)[:-1]

    lines = [LineString([points[i], points[j]]) for i, j in boundary_edges]
    polys = list(polygonize(unary_union(lines)))

    if not polys:
        hull = MultiPoint(points).convex_hull
        return np.asarray(hull.exterior.coords)[:-1], [np.asarray(hull.exterior.coords)[:-1]]  

    #返回全部的polys 
    all_coords = [np.asarray(p.exterior.coords)[:-1] for p in polys]  

    poly = max(polys, key=lambda p: p.area)
    coords = np.asarray(poly.exterior.coords)[:-1]
    return coords, all_coords  


def concave_boundary_raster(pts,
                            pixel=None,
                            pad=10,
                            close_r=3,
                            sigma=1.5,
                            level=0.5):
    """
    栅格化凹包：落点 -> 形态学闭运算 -> 高斯平滑 -> 等值线提取
    返回:
      boundary: 最长外轮廓 (M, 2)
      all_boundaries: 所有轮廓列表
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None, None

    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    span = max(xmax - xmin, ymax - ymin)
    if pixel is None:
        pixel = span / 400.0 if span > 0 else 1.0
    if pixel <= 0:
        pixel = 1.0

    W = int(np.ceil((xmax - xmin) / pixel)) + 1 + 2 * pad
    H = int(np.ceil((ymax - ymin) / pixel)) + 1 + 2 * pad
    W = max(W, 3)
    H = max(H, 3)

    mask = np.zeros((H, W), dtype=bool)
    ix = np.round((x - xmin) / pixel).astype(int) + pad
    iy = np.round((y - ymin) / pixel).astype(int) + pad
    ix = np.clip(ix, 0, W - 1)
    iy = np.clip(iy, 0, H - 1)
    mask[iy, ix] = True

    if close_r and close_r > 0:
        mask = binary_closing(mask, disk(int(close_r)))

    img = mask.astype(float)
    if sigma and sigma > 0:
        img = gaussian_filter(img, sigma=float(sigma))

    contours = find_contours(img, level=float(level))
    if not contours:
        return None, None

    all_boundaries = []
    for c in contours:
        boundary_i = np.column_stack([
            (c[:, 1] - pad) * pixel + xmin,
            (c[:, 0] - pad) * pixel + ymin,
        ])
        if len(boundary_i) >= 3:
            all_boundaries.append(boundary_i)

    if not all_boundaries:
        return None, None

    boundary = max(all_boundaries, key=len)
    return boundary, all_boundaries

# 自动检测分隔符
def detect_delimiter(content):
    """
    自动检测文件分隔符
    """
    delimiters = [',', '\t', ';', '|', ' ']
    for delimiter in delimiters:
        try:
            df = pd.read_csv(io.StringIO(content), sep=delimiter, nrows=5)
            if len(df.columns) >= 2:
                return delimiter
        except:
            continue
    return ','

# 解析上传的文件
def parse_contents(contents, filename):
    """
    解析上传的文件内容
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        content = decoded.decode('utf-8')
        delimiter = detect_delimiter(content)
        
        if filename.endswith('.csv') or filename.endswith('.xls'):
            df = pd.read_csv(io.StringIO(content), sep=delimiter)
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.txt'):
            df = pd.read_csv(io.StringIO(content), sep=delimiter)
        else:
            return None
        
        return df.dropna()
    
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# 创建空白图表的函数，确保没有网格线，白色背景
def create_empty_figure():
    """
    创建空白图表，没有网格线和白色背景
    """
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=False
        ),
        hovermode=False,
        showlegend=False,
        dragmode='pan',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig



# 布局
app.layout = dbc.Container([
    # 顶部标题区域
    dbc.Row([
        dbc.Col([
            html.Div([
                # Logo 和标题组合
                html.Div([
                    html.Div([
                        html.H1("点云轮廓可视化", className="text-center", 
                               style={'color': '#1d1d1f', 'font-weight': '700', 'margin': '0', 'line-height': '1.2'}),
                        html.P("基于凹包和凸包算法的智能轮廓提取系统",
                              className="text-center",
                              style={'color': '#86868b', 'font-size': '17px', 'font-weight': '400', 'margin': '8px 0 0 0'})
                    ], style={'flex': '1'})
                ], style={
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center',
                    'gap': '20px',
                    'flex-wrap': 'wrap'
                })
            ], style={'padding': '40px 0 20px 0'})
        ])
    ]),
    
    # 文件上传区域
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div("📄", style={'display': 'inline-block', 'margin-right': '10px', 'font-size': '24px'}),
                        html.H5("上传数据文件", style={'display': 'inline-block', 'margin': '0'})
                    ], style={'margin-bottom': '20px'}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            # 图标区域
                            html.Div([
                                html.Div("📁", className="upload-icon"),
                            ], style={'margin-bottom': '20px'}),
                            
                            # 主标题
                            html.Div('拖拽文件到此处', className="upload-title"),
                            
                            # 分隔线
                            html.Div([
                                html.Div(style={'flex': '1', 'height': '1px', 'background': '#d1d1d6'}),
                                html.Span('或', style={'padding': '0 16px', 'color': '#86868b', 'font-size': '14px'}),
                                html.Div(style={'flex': '1', 'height': '1px', 'background': '#d1d1d6'})
                            ], style={'display': 'flex', 'align-items': 'center', 'margin': '16px 0'}),
                            
                            # 点击上传按钮
                            html.Div([
                                html.Span('点击选择文件', className="upload-button")
                            ], style={'margin-bottom': '20px'}),
                            
                            # 支持格式说明
                            html.Div([
                                html.Div('📋 支持格式', style={
                                    'font-size': '13px', 
                                    'font-weight': '600', 
                                    'color': '#1d1d1f',
                                    'margin-bottom': '8px'
                                }),
                                html.Div([
                                    html.Span('CSV', className='format-tag'),
                                    html.Span('XLS', className='format-tag'),
                                    html.Span('XLSX', className='format-tag'),
                                    html.Span('TXT', className='format-tag')
                                ], style={'display': 'flex', 'gap': '8px', 'justify-content': 'center', 'flex-wrap': 'wrap'})
                            ])
                        ], className="upload-area"),
                        multiple=False
                    ),
                    html.Div(id='upload-status', className='mt-3')
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # 主要内容区域
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # 数据列选择部分
                    html.Div([
                        html.Div([
                            html.Div("⚙️", style={'display': 'inline-block', 'margin-right': '10px', 'font-size': '24px'}),
                            html.H5("数据配置", style={'display': 'inline-block', 'margin': '0'})
                        ], style={'margin-bottom': '20px'}),
                        html.Div("数据列映射", className="section-title"),
                        
                        html.Label("X轴列"),
                        dcc.Dropdown(
                            id='x-axis-dropdown',
                            options=[],
                            placeholder='选择X轴数据列',
                            className='dash-dropdown',
                            style={'margin-bottom': '16px'}
                        ),
                        
                        html.Label("Y轴列"),
                        dcc.Dropdown(
                            id='y-axis-dropdown',
                            options=[],
                            placeholder='选择Y轴数据列',
                            className='dash-dropdown',
                            style={'margin-bottom': '16px'}
                        ),
                        
                        html.Label("颜色分组列（可选）"),
                        html.Div("用于区分不同数据组，留空则使用统一颜色", className="hint-text"),
                        dcc.Dropdown(
                            id='color-dropdown',
                            options=[],
                            placeholder='不选择则全部使用灰色',
                            className='dash-dropdown'
                        ),
                    ]),
                    
                    # Apple 风格分割线
                    html.Div(className="section-divider"),
                    
                    # 轮廓计算设置部分
                    html.Div([
                        html.Div("轮廓计算设置", className="section-title"),
                        
                        html.Label("计算方法"),
                        dcc.Dropdown(
                            id='method-dropdown',
                            options=[
                                {'label': '🔷 最小凸包 - 简单快速', 'value': 'convex'},
                                {'label': '🔶 最小凹包 - 精确贴合', 'value': 'concave'},
                                {'label': '🟠 栅格凹包 - 平滑边界', 'value': 'concave_raster'}
                            ],
                            value='convex',
                            className='dash-dropdown',
                            style={'margin-bottom': '20px'}
                        ),
                        
                        html.Div([
                            html.Label("Alpha 参数（凹包专用）"),
                            html.Div(
                                "控制轮廓的贴合程度：值越大越贴近数据点，值越小越平滑",
                                className="hint-text",
                                id='alpha-hint-text'
                            ),
                            dcc.Slider(
                                id='alpha-slider',
                                min=0.01,
                                max=1.0,
                                step=0.01,
                                value=0.1,
                                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], id='alpha-controls-wrapper', style={'display': 'none'}),

                        html.Div([
                            html.Div("栅格凹包参数（栅格凹包专用）", className="section-title"),
                            html.Div(
                                "pixel 留空表示自动估计；其余参数可按数据密度调整",
                                className="hint-text",
                                id='raster-hint-text'
                            ),

                            html.Label("Pixel 像素大小（可选）"),
                            dcc.Input(
                                id='raster-pixel-input',
                                type='number',
                                value=None,
                                placeholder='留空自动估计',
                                debounce=True,
                                className='dash-dropdown',
                                style={'margin-bottom': '12px', 'width': '100%'}
                            ),

                            html.Label("Close 半径"),
                            dcc.Slider(
                                id='raster-close-r-slider',
                                min=0,
                                max=30,
                                step=1,
                                value=10,
                                marks={0: '0', 5: '5', 10: '10', 20: '20', 30: '30'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            html.Label("Sigma 平滑"),
                            dcc.Slider(
                                id='raster-sigma-slider',
                                min=0,
                                max=10,
                                step=0.1,
                                value=2.0,
                                marks={0: '0', 1: '1', 2: '2', 5: '5', 10: '10'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            html.Label("Level 等值线阈值"),
                            dcc.Slider(
                                id='raster-level-slider',
                                min=0.05,
                                max=0.95,
                                step=0.01,
                                value=0.5,
                                marks={0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], id='raster-controls-wrapper', style={'display': 'none'}),
                        
                        html.Div([
                            dbc.Button([
                                html.Span("✨ ", style={'margin-right': '6px'}),
                                "计算轮廓"
                            ], id="compute-btn", 
                               className="btn-custom btn-primary-custom mt-3 me-2",
                               style={'min-width': '140px'}),
                            dbc.Button([
                                html.Span("🔗 ", style={'margin-right': '6px'}),
                                "计算交集"
                            ], id="intersection-btn", 
                               className="btn-custom btn-primary-custom mt-3",
                               style={'min-width': '140px'})
                        ], style={'display': 'flex', 'gap': '10px', 'flex-wrap': 'wrap'}),
                        
                        html.Div(id='compute-status', className='mt-3', style={
                            'padding': '12px 16px',
                            'border-radius': '10px',
                            'background': 'linear-gradient(135deg, #e8f4fd 0%, #d4e9fc 100%)',
                            'display': 'none',
                            'align-items': 'center',
                            'gap': '10px'
                        })
                    ])
                ])
            ], style={'height': '100%'})
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("点云/轮廓可视化", style={'margin': '0 0 16px 0'})
                    ]),
                    dcc.Loading(
                        id="loading-graph",
                        type="default",
                        children=[
                            dcc.Graph(id='point-cloud-graph', 
                                    config={
                                        'displayModeBar': True,
                                        'scrollZoom': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                                    },
                                    style={'height': '600px', 'border-radius': '14px'})
                        ],
                        style={'border-radius': '14px'},
                        color='#007aff'
                    )
                ])
            ])
        ], width=8)
    ], className="mb-4"),

    # 所有轮廓可视化区域
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("所有轮廓可视化", style={'margin': '0 0 16px 0'})
                    ]),
                    dcc.Loading(
                        id="loading-all-contours-graph",
                        type="default",
                        children=[
                            dcc.Graph(id='all-contours-graph',
                                    config={
                                        'displayModeBar': True,
                                        'scrollZoom': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                                    },
                                    style={'height': '500px', 'border-radius': '14px'})
                        ],
                        style={'border-radius': '14px'},
                        color='#007aff'
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # 交集轮廓可视化区域
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("轮廓交集可视化", style={'margin': '0 0 8px 0'}),
                        html.Div([
                            html.Span("💡 提示: ", style={'font-weight': '600', 'color': '#007aff'}),
                            html.Span("计算多个分组轮廓的交集区域")
                        ], style={'color': '#86868b', 'font-size': '14px', 'margin-bottom': '16px'})
                    ]),
                    html.Div(id='intersection-status', className='mb-3'),
                    dbc.Button([
                        html.Span("📥 ", style={'margin-right': '6px'}),
                        "添加ROI并下载"
                    ], id="download-roi-btn", 
                       className="btn-custom btn-primary-custom mb-3",
                       style={'min-width': '160px'}),
                    dcc.Loading(
                        id="loading-intersection-graph",
                        type="default",
                        children=[
                            dcc.Graph(id='intersection-graph', 
                                    config={
                                        'displayModeBar': True,
                                        'scrollZoom': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                                    },
                                    style={'height': '500px', 'border-radius': '14px'})
                        ],
                        style={'border-radius': '14px'},
                        color='#007aff'
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4", id='intersection-row', style={'display': 'none'}),
    
    # 存储组件
    dcc.Store(id='data-store'),
    dcc.Store(id='contours-store'),  # 存储计算的轮廓数据
    dcc.Store(id='intersection-store'),  # 存储交集区域数据
    dcc.Store(id='computing-state', data=False),
    dcc.Download(id="download-dataframe-csv"),  # 下载组件
    
    # 页脚
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("© 2026 点云轮廓计算工具 · 基于 Dash & Plotly 构建",
                      style={'color': '#86868b', 'font-size': '13px', 'margin': '0', 'text-align': 'center'})
            ], style={'padding': '30px 0 20px 0'})
        ])
    ])
    
], fluid=True, style={'padding': '0 40px', 'max-width': '1600px', 'margin': '0 auto'})

# 回调：处理文件上传
@app.callback(
    [Output('data-store', 'data'),
     Output('upload-status', 'children'),
     Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('color-dropdown', 'options'),
     Output('x-axis-dropdown', 'value'),
     Output('y-axis-dropdown', 'value')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_data(contents, filename):
    if contents is None:
        return None, "", [], [], [], None, None
    
    df = parse_contents(contents, filename)
    
    if df is None:
        return None, dbc.Alert("文件解析失败，请检查文件格式", color="danger"), [], [], [], None, None
    
    global current_data
    current_data = df
    
    # 创建列选项
    column_options = [{'label': col, 'value': col} for col in df.columns]
    
    # 自动检测默认的x, y列
    x_default = None
    y_default = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in ['x', 'x坐标', 'longitude', 'lon'] and x_default is None:
            x_default = col
        elif col_lower in ['y', 'y坐标', 'latitude', 'lat'] and y_default is None:
            y_default = col
    
    # 如果没找到，使用前两列
    if x_default is None and len(df.columns) >= 1:
        x_default = df.columns[0]
    if y_default is None and len(df.columns) >= 2:
        y_default = df.columns[1]
    
    return (df.to_json(), 
            dbc.Alert(f"成功加载 {len(df)} 个数据点，{len(df.columns)} 列", color="success"),
            column_options, column_options, column_options,
            x_default, y_default)


# 回调：根据列选择更新图表
@app.callback(
    Output('point-cloud-graph', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('color-dropdown', 'value')],
    State('data-store', 'data')
)
def update_graph_from_columns(x_col, y_col, color_col, data_json):
    if not data_json or not x_col or not y_col:
        return create_empty_figure()
    
    df = pd.read_json(io.StringIO(data_json))
    fig = go.Figure()
    
    # 对于大数据集使用 Scattergl (WebGL渲染) 而不是 Scatter (SVG渲染)
    scatter_type = go.Scattergl if len(df) > 1000 else go.Scatter
    
    if color_col and color_col in df.columns:
        # 按颜色列分组
        groups = df.groupby(color_col)
        colors = ['#007aff', '#ff3b30', '#34c759', '#ff9500', '#af52de', 
                  '#ff2d55', '#5ac8fa', '#ffcc00', '#ff6482', '#64d2ff']
        
        for i, (group_name, group_data) in enumerate(groups):
            fig.add_trace(scatter_type(
                x=group_data[x_col],
                y=group_data[y_col],
                mode='markers',
                marker=dict(size=5, color=colors[i % len(colors)], opacity=0.6),
                name=f'{color_col}={group_name}'
            ))
    else:
        # 全部使用灰色
        fig.add_trace(scatter_type(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(size=5, color='#8e8e93', opacity=0.6),
            name='点云数据'
        ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, scaleanchor='y', scaleratio=1, title=x_col, fixedrange=False, zeroline=False),
        yaxis=dict(showgrid=False, title=y_col, fixedrange=False, zeroline=False),
        hovermode='closest',
        showlegend=True,
        dragmode='pan'
    )
    
    return fig



# 回调：显示计算状态
@app.callback(
    Output('compute-status', 'children'),
    Output('compute-status', 'style'),
    Input('compute-btn', 'n_clicks'),
    prevent_initial_call=True
)
def show_computing_status(n_clicks):
    if n_clicks:
        return [
            html.Div("⚙️", style={'font-size': '18px', 'animation': 'spin 1s linear infinite'}),
            html.Span("正在计算轮廓，请稍候...", style={'color': '#007aff', 'font-weight': '500', 'font-size': '14px'})
        ], {
            'padding': '12px 16px',
            'border-radius': '10px',
            'background': 'linear-gradient(135deg, #e8f4fd 0%, #d4e9fc 100%)',
            'display': 'flex',
            'align-items': 'center',
            'gap': '10px',
            'margin-top': '16px'
        }
    return "", {'display': 'none'}


@app.callback(
    [Output('alpha-controls-wrapper', 'style'),
     Output('raster-controls-wrapper', 'style'),
     Output('raster-pixel-input', 'disabled'),
     Output('raster-close-r-slider', 'disabled'),
     Output('raster-sigma-slider', 'disabled'),
     Output('raster-level-slider', 'disabled'),
     Output('raster-hint-text', 'children'),
     Output('alpha-slider', 'disabled'),
     Output('alpha-hint-text', 'children')],
    Input('method-dropdown', 'value')
)
def toggle_raster_controls(method):
    is_raster = method == 'concave_raster'
    is_alpha = method == 'concave'

    alpha_style = {'display': 'block'} if is_alpha else {'display': 'none'}
    raster_style = {'display': 'block'} if is_raster else {'display': 'none'}

    disabled = not is_raster
    if is_raster:
        hint = "pixel 留空表示自动估计；其余参数可按数据密度调整"
    else:
        hint = "当前方法不使用栅格参数（切换到“栅格凹包”后可编辑）"

    alpha_disabled = not is_alpha
    if is_alpha:
        alpha_hint = "控制轮廓的贴合程度：值越大越贴近数据点，值越小越平滑"
    else:
        alpha_hint = "当前方法不使用 Alpha 参数（切换到“最小凹包”后可编辑）"

    return alpha_style, raster_style, disabled, disabled, disabled, disabled, hint, alpha_disabled, alpha_hint



# 回调：计算轮廓
@app.callback(
    [Output('point-cloud-graph', 'figure', allow_duplicate=True),
    Output('contours-store', 'data'),
    Output('all-contours-graph', 'figure')],
    Input('compute-btn', 'n_clicks'),
    [State('data-store', 'data'),
     State('method-dropdown', 'value'),
     State('alpha-slider', 'value'),
     State('raster-pixel-input', 'value'),
     State('raster-close-r-slider', 'value'),
     State('raster-sigma-slider', 'value'),
     State('raster-level-slider', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('color-dropdown', 'value')],
    prevent_initial_call=True
)
def compute_contour(n_clicks, data_json, method, alpha,
                    raster_pixel, raster_close_r, raster_sigma, raster_level,
                    x_col, y_col, color_col):
    if not data_json or not x_col or not y_col:
        return create_empty_figure(), None, create_empty_figure()
    
    df = pd.read_json(io.StringIO(data_json))
    fig = go.Figure()
    all_contours_fig = go.Figure()
    
    # 存储轮廓数据用于交集计算
    contours_data = []
    
    # 对于大数据集使用 Scattergl (WebGL渲染)
    scatter_type = go.Scattergl if len(df) > 1000 else go.Scatter
    
    # 颜色配置
    colors = ['#007aff', '#ff3b30', '#34c759', '#ff9500', '#af52de', 
              '#ff2d55', '#5ac8fa', '#ffcc00', '#ff6482', '#64d2ff']
    all_contour_colors = [
        '#007aff', '#ff3b30', '#34c759', '#ff9500', '#af52de',
        '#ff2d55', '#5ac8fa', '#ffcc00', '#64d2ff', '#30d158',
        '#bf5af2', '#ffd60a', '#0a84ff', '#ff9f0a', '#ff375f'
    ]
    
    def hex_to_rgba(hex_color, alpha):
        """将十六进制颜色转换为rgba"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    def get_contour_points(points, method_value, alpha_value,
                           pixel_value, close_r_value, sigma_value, level_value):
        if method_value == 'convex':
            hull = ConvexHull(points)
            contour = points[hull.vertices]
            return contour, [contour]

        if method_value == 'concave':
            contour_coords, all_coords = alpha_shape_algo(points, alpha_value)
            if contour_coords is not None and len(contour_coords) >= 3:
                contour = np.array(contour_coords)
                return contour, all_coords

        if method_value == 'concave_raster':
            safe_close_r = 10 if close_r_value is None else max(0, int(close_r_value))
            safe_sigma = 2.0 if sigma_value is None else max(0.0, float(sigma_value))
            safe_level = 0.5 if level_value is None else float(level_value)
            safe_level = min(0.99, max(0.01, safe_level))
            safe_pixel = None
            if pixel_value is not None:
                px = float(pixel_value)
                safe_pixel = px if px > 0 else None
            boundary, all_boundaries = concave_boundary_raster(
                points,
                pixel=safe_pixel,
                close_r=safe_close_r,
                sigma=safe_sigma,
                level=safe_level
            )
            if boundary is not None and len(boundary) >= 3:
                valid_boundaries = []
                if all_boundaries:
                    for contour in all_boundaries:
                        contour_arr = np.asarray(contour)
                        if contour_arr.ndim == 2 and contour_arr.shape[1] == 2 and len(contour_arr) >= 3:
                            valid_boundaries.append(contour_arr)
                if not valid_boundaries:
                    valid_boundaries = [boundary]
                return boundary, valid_boundaries

        hull = ConvexHull(points)
        contour = points[hull.vertices]
        return contour, [contour]
    
    try:
        if color_col and color_col in df.columns:
            # 按组分别计算轮廓
            groups = df.groupby(color_col)
            
            for i, (group_name, group_data) in enumerate(groups):
                # 绘制点云
                fig.add_trace(scatter_type(
                    x=group_data[x_col],
                    y=group_data[y_col],
                    mode='markers',
                    marker=dict(size=5, color=colors[i % len(colors)], opacity=0.6),
                    name=f'{color_col}={group_name}',
                    showlegend=True
                ))

                all_contours_fig.add_trace(scatter_type(
                    x=group_data[x_col],
                    y=group_data[y_col],
                    mode='markers',
                    marker=dict(size=4, color=colors[i % len(colors)], opacity=0.45),
                    name=f'{color_col}={group_name}',
                    showlegend=True
                ))
                
                # 计算该组的轮廓
                points = group_data[[x_col, y_col]].values
                
                if len(points) >= 3:
                    contour_points, all_group_contours = get_contour_points(
                        points,
                        method,
                        alpha,
                        raster_pixel,
                        raster_close_r,
                        raster_sigma,
                        raster_level
                    )
                    
                    # 保存轮廓数据
                    contours_data.append({
                        'name': str(group_name),
                        'points': contour_points.tolist(),
                        'color': colors[i % len(colors)]
                    })
                    
                    # 绘制轮廓
                    contour_x = list(contour_points[:, 0]) + [contour_points[0, 0]]
                    contour_y = list(contour_points[:, 1]) + [contour_points[0, 1]]
                    
                    fig.add_trace(go.Scatter(
                        x=contour_x,
                        y=contour_y,
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=2),
                        fill='toself',
                        fillcolor=hex_to_rgba(colors[i % len(colors)], 0.1),
                        name=f'轮廓-{group_name}',
                        showlegend=True
                    ))

                    for k, contour_item in enumerate(all_group_contours):
                        contour_item = np.asarray(contour_item)
                        contour_item_x = list(contour_item[:, 0]) + [contour_item[0, 0]]
                        contour_item_y = list(contour_item[:, 1]) + [contour_item[0, 1]]
                        contour_color = all_contour_colors[(i * 7 + k) % len(all_contour_colors)]
                        all_contours_fig.add_trace(go.Scatter(
                            x=contour_item_x,
                            y=contour_item_y,
                            mode='lines',
                            line=dict(color=contour_color, width=1.8, dash='dot'),
                            name=f'全部轮廓-{group_name}-{k+1}',
                            showlegend=True
                        ))
        else:
            # 不分组，统一计算
            points = df[[x_col, y_col]].values
            
            # 绘制点云
            fig.add_trace(scatter_type(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(size=5, color='#8e8e93', opacity=0.6),
                name='点云数据'
            ))

            all_contours_fig.add_trace(scatter_type(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(size=5, color='#8e8e93', opacity=0.5),
                name='点云数据'
            ))
            
            # 计算轮廓
            contour_points, all_group_contours = get_contour_points(
                points,
                method,
                alpha,
                raster_pixel,
                raster_close_r,
                raster_sigma,
                raster_level
            )
            
            # 保存轮廓数据（无分组情况）
            contours_data.append({
                'name': '统一轮廓',
                'points': contour_points.tolist(),
                'color': '#34c759'
            })
            
            # 绘制轮廓
            contour_x = list(contour_points[:, 0]) + [contour_points[0, 0]]
            contour_y = list(contour_points[:, 1]) + [contour_points[0, 1]]
            
            fig.add_trace(go.Scatter(
                x=contour_x,
                y=contour_y,
                mode='lines',
                line=dict(color='#34c759', width=2),
                fill='toself',
                fillcolor='rgba(52, 199, 89, 0.1)',
                name='轮廓'
            ))

            for k, contour_item in enumerate(all_group_contours):
                contour_item = np.asarray(contour_item)
                contour_item_x = list(contour_item[:, 0]) + [contour_item[0, 0]]
                contour_item_y = list(contour_item[:, 1]) + [contour_item[0, 1]]
                contour_color = all_contour_colors[k % len(all_contour_colors)]
                all_contours_fig.add_trace(go.Scatter(
                    x=contour_item_x,
                    y=contour_item_y,
                    mode='lines',
                    line=dict(color=contour_color, width=1.8, dash='dot'),
                    name=f'全部轮廓-{k+1}',
                    showlegend=True
                ))
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, scaleanchor='y', scaleratio=1, title=x_col, fixedrange=False, zeroline=False),
            yaxis=dict(showgrid=False, title=y_col, fixedrange=False, zeroline=False),
            hovermode='closest',
            showlegend=True,
            dragmode='pan'
        )

        all_contours_fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, scaleanchor='y', scaleratio=1, title=x_col, fixedrange=False, zeroline=False),
            yaxis=dict(showgrid=False, title=y_col, fixedrange=False, zeroline=False),
            hovermode='closest',
            showlegend=True,
            dragmode='pan'
        )
        
        return fig, contours_data, all_contours_fig
    
    except Exception as e:
        print(f"Error computing contour: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure(), None, go.Figure()

# 回调：计算交集
@app.callback(
    [Output('intersection-graph', 'figure'),
     Output('intersection-status', 'children'),
     Output('intersection-row', 'style'),
     Output('intersection-store', 'data')],
    Input('intersection-btn', 'n_clicks'),
    [State('contours-store', 'data'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value')],
    prevent_initial_call=True
)
def compute_intersection(n_clicks, contours_data, x_col, y_col):
    if not contours_data or len(contours_data) < 2:
        return (create_empty_figure(), 
                dbc.Alert("请先计算至少2个分组的轮廓才能计算交集！", color="warning"),
                {'display': 'none'},
                None)
    
    try:
        # 创建Polygon对象
        polygons = []
        for contour in contours_data:
            points = np.array(contour['points'])
            polygon = Polygon(points)
            polygons.append(polygon)
        
        # 计算交集
        intersection = polygons[0]
        for poly in polygons[1:]:
            intersection = intersection.intersection(poly)
        
        # 检查交集是否为空
        if intersection.is_empty:
            return (create_empty_figure(),
                    dbc.Alert("这些轮廓没有交集区域！", color="info"),
                    {'display': 'block'},
                    None)
        
        # 创建图表
        fig = go.Figure()
        
        # 颜色配置
        colors = ['#007aff', '#ff3b30', '#34c759', '#ff9500', '#af52de', 
                  '#ff2d55', '#5ac8fa', '#ffcc00', '#ff6482', '#64d2ff']
        
        def hex_to_rgba(hex_color, alpha):
            """将十六进制颜色转换为rgba"""
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r}, {g}, {b}, {alpha})'
        
        # 绘制原始轮廓
        for i, contour in enumerate(contours_data):
            points = np.array(contour['points'])
            contour_x = list(points[:, 0]) + [points[0, 0]]
            contour_y = list(points[:, 1]) + [points[0, 1]]
            
            color = contour.get('color', colors[i % len(colors)])
            
            fig.add_trace(go.Scatter(
                x=contour_x,
                y=contour_y,
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f"轮廓-{contour['name']}",
                showlegend=True
            ))
        
        # 绘制交集区域
        if hasattr(intersection, 'exterior'):
            # 单个多边形交集
            coords = list(intersection.exterior.coords)
            intersection_points = np.array(coords)
            intersection_x = list(intersection_points[:, 0])
            intersection_y = list(intersection_points[:, 1])
            
            fig.add_trace(go.Scatter(
                x=intersection_x,
                y=intersection_y,
                mode='lines',
                line=dict(color='#ff3b30', width=3),
                fill='toself',
                fillcolor='rgba(255, 59, 48, 0.3)',
                name='交集区域',
                showlegend=True
            ))
            
            # 计算交集面积
            area = intersection.area
            status_message = dbc.Alert([
                html.Strong("✅ 成功计算交集！"),
                html.Br(),
                f"交集面积: {area:.2f} 平方单位"
            ], color="success")
            
            # 保存交集数据供ROI使用
            intersection_data = {
                'coords': coords,
                'area': area
            }
        else:
            # 多个不连续的交集区域
            status_message = dbc.Alert("交集包含多个不连续区域", color="info")
            intersection_data = None
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, scaleanchor='y', scaleratio=1, title=x_col, fixedrange=False, zeroline=False),
            yaxis=dict(showgrid=False, title=y_col, fixedrange=False, zeroline=False),
            hovermode='closest',
            showlegend=True,
            dragmode='pan'
        )
        
        return fig, status_message, {'display': 'block'}, intersection_data
    
    except Exception as e:
        print(f"Error computing intersection: {e}")
        import traceback
        traceback.print_exc()
        return (create_empty_figure(),
                dbc.Alert(f"计算交集时出错: {str(e)}", color="danger"),
                {'display': 'block'},
                None)

# 回调：下载ROI数据
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-roi-btn", "n_clicks"),
    [State('data-store', 'data'),
     State('intersection-store', 'data'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value')],
    prevent_initial_call=True
)
def download_roi_data(n_clicks, data_json, intersection_data, x_col, y_col):
    if not data_json or not intersection_data or not x_col or not y_col:
        return None
    
    try:
        # 读取原始数据
        df = pd.read_json(io.StringIO(data_json))
        
        # 创建交集多边形
        coords = intersection_data['coords']
        intersection_polygon = Polygon(coords)
        
        # 判断每个点是否在交集区域内
        inhull = []
        for idx, row in df.iterrows():
            point = Point(row[x_col], row[y_col])
            inhull.append(intersection_polygon.contains(point))
        
        # 添加inhull列
        df['inhull'] = inhull
        
        # 统计信息
        points_in_hull = sum(inhull)
        print(f"Total points: {len(df)}, Points in hull: {points_in_hull}")
        
        # 生成CSV文件
        return dcc.send_data_frame(df.to_csv, "data_with_roi.csv", index=False)
    
    except Exception as e:
        print(f"Error generating ROI data: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    app.run(debug=True, port=8050)
