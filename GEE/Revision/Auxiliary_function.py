import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from tqdm import tqdm
from pathlib import Path
from pyproj import Transformer
from pathlib import Path
import Auxiliary_function as F
import pandas as pd
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde



def load_refer_icesat_point_optimized_shp(
    FB_raster: str, 
    icesat_csv_path: str, 
    output_point: str, 
    output_raw_point: str,
    lon_col: str = 'mean_lon', 
    lat_col: str = 'mean_lat',
    oh_threshold: float = 5.0, 
    b8_min: float = 0.0
) -> gpd.GeoDataFrame:
    """
    加载 ICESat-2 点数据，提取栅格特征，并进行质量筛选。
    输出格式为 ESRI Shapefile (.shp)。

    Args:
        FB_raster (str): 多波段光学影像 GeoTIFF 文件路径。
        icesat_csv_path (str): ICESat-2 点数据 CSV 文件路径。
        output_point (str): 筛选后的 GeoDataFrame 输出路径 (需以 .shp 结尾)。
        output_raw_point (str): 原始 GeoDataFrame 输出路径 (需以 .shp 结尾)。
        lon_col (str): CSV 中经度列的名称。
        lat_col (str): CSV 中纬度列的名称。
        oh_threshold (float): OH (高程观测精度) 的最大阈值。
        b8_min (float): b8 波段的最小值筛选阈值。

    Returns:
        gpd.GeoDataFrame: 筛选后的 ICESat-2 GeoDataFrame。
    """
    
    print(f"--- 正在加载和处理 ICESat-2 点数据 ---")
    
    # 1. 加载 CSV 并转换为 GeoDataFrame
    df = pd.read_csv(icesat_csv_path)
    
    # 转换为 shapely Point 对象
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # 明确设置原始 CRS 为 WGS 84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    points = gdf.copy() 
    
    # 2. 提取栅格特征
    with rasterio.open(FB_raster) as src:
        print(f"GeoTIFF CRS: {src.crs}. 原始点 CRS: {points.crs}")
        
        # 坐标系转换：如果栅格和点的 CRS 不一致，进行转换
        if points.crs != src.crs:
            print(f"执行 CRS 转换: {points.crs} -> {src.crs}")
            points = points.to_crs(src.crs)
            
        # 提取坐标列表 (兼容新旧 geopandas 版本)
        coords = [(point.x, point.y) for point in points.geometry]

        # 批量采样：使用 numpy 数组进行高性能操作
        print(f"开始从 {src.count} 个波段中批量采样...")
        try:
            # V 是一个包含所有点所有波段值的 N x C NumPy 数组
            V = np.array(list(src.sample(coords, masked=False)))
        except Exception as e:
            print(f"采样失败，请检查点是否落在栅格范围内或栅格完整性。错误: {e}")
            return None 
        
        band_count = src.count
        
        # 将采样值添加到 GeoDataFrame
        for band in range(band_count):
            # 直接从 NumPy 数组中提取列，高效地赋给 GeoDataFrame
            # ⚠️ 警告：字段名 'bX_FB6' 超过 10 个字符，在 Shapefile 中会被截断！
            points[f"b{band+1}_FB6"] = V[:, band] 
            
        if f"b8_FB6" not in points.columns:
             print("警告: b8_FB6 列可能不存在，请检查栅格波段数量或索引。")
    
    # 3. 数据筛选（假设 'OH' 列存在）
    
    # 构造筛选掩码
    if 'OH' in points.columns and f'b8_FB6' in points.columns:
        icesat_mask = (points['OH'] < oh_threshold) & (points['b8_FB6'] > b8_min)
        data = points[icesat_mask].copy() 
        print(f"原始点数量: {len(points)}，筛选后点数量: {len(data)}")
    else:
        print("警告: 缺少 'OH' 或 'b8_FB6' 列，跳过筛选。")
        data = points.copy()
        
    # 4. 文件保存 - 使用 ESRI Shapefile Driver
    
    try:
        # 指定 driver='ESRI Shapefile'
        points.to_file(output_raw_point, driver='ESRI Shapefile')
        data.to_file(output_point, driver='ESRI Shapefile')
        
        print(f"\n原始点数据已成功保存至: {output_raw_point} (Shapefile)")
        print(f"筛选点数据已成功保存至: {output_point} (Shapefile)")
        
    except Exception as e:
        print(f"\n--- 致命错误：Shapefile 保存失败 ---")
        print(f"错误信息: {e}")
        print("💡 **请注意：** Shapefile 字段名不能超过 10 个字符！请检查列名。")
        
    return data

def load_refer_icesat_point_optimized_shp2(
    FB_raster: str, 
    shp_path: str, 
    output_point: str, 
    lon_col: str = 'mean_lon', 
    lat_col: str = 'mean_lat',
    oh_threshold: float = 5.0, 
    b8_min: float = 0.0
) -> gpd.GeoDataFrame:
    
    # if shp_path
    # df = gpd.read_file(shp_path)
    df = pd.read_csv(shp_path)
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df[['Unnamed_ 0', 'OH', 'mean_heigh','mean_lon', 'mean_lat', 'Strip_No']], geometry=geometry)

    gdf.set_crs(epsg=4326, inplace=True)
    points = gdf.copy()
    
    with rasterio.open(FB_raster) as src:
        print(f"GeoTIFF CRS: {src.crs}. 原始点 CRS: {points.crs}")
        
        # 坐标系转换：如果栅格和点的 CRS 不一致，进行转换
        if points.crs != src.crs:
            print(f"执行 CRS 转换: {points.crs} -> {src.crs}")
            points = points.to_crs(src.crs)
            
        # 提取坐标列表 (兼容新旧 geopandas 版本)
        coords = [(point.x, point.y) for point in points.geometry]

        # 批量采样：使用 numpy 数组进行高性能操作
        print(f"开始从 {src.count} 个波段中批量采样...")
        try:
            # V 是一个包含所有点所有波段值的 N x C NumPy 数组
            V = np.array(list(src.sample(coords, masked=False)))
        except Exception as e:
            print(f"采样失败，请检查点是否落在栅格范围内或栅格完整性。错误: {e}")
            return None 
        
        band_count = src.count
        
        # 将采样值添加到 GeoDataFrame
        for band in range(band_count):
            # 直接从 NumPy 数组中提取列，高效地赋给 GeoDataFrame
            # ⚠️ 警告：字段名 'bX_FB6' 超过 10 个字符，在 Shapefile 中会被截断！
            points[f"b{band+1}_FB6"] = V[:, band] 
            
        if f"b8_FB6" not in points.columns:
             print("警告: b8_FB6 列可能不存在，请检查栅格波段数量或索引。")
    
    # 3. 数据筛选（假设 'OH' 列存在）
    
    # 构造筛选掩码
    if 'OH' in points.columns and f'b8_FB6' in points.columns:
        icesat_mask = (points['OH'] < oh_threshold) & (points['b8_FB6'] > b8_min)
        data = points[icesat_mask].copy() 
        print(f"原始点数量: {len(points)}，筛选后点数量: {len(data)}")
    else:
        print("警告: 缺少 'OH' 或 'b8_FB6' 列，跳过筛选。")
        data = points.copy()
        
    # 4. 文件保存 - 使用 ESRI Shapefile Driver
    
    try:
        # 指定 driver='ESRI Shapefile'
        # points.to_file(output_raw_point, driver='ESRI Shapefile')
        data.to_file(output_point, driver='ESRI Shapefile')
        
        # print(f"\n原始点数据已成功保存至: {output_raw_point} (Shapefile)")
        print(f"筛选点数据已成功保存至: {output_point} (Shapefile)")
        
    except Exception as e:
        print(f"\n--- 致命错误：Shapefile 保存失败 ---")
        print(f"错误信息: {e}")
        print("💡 **请注意：** Shapefile 字段名不能超过 10 个字符！请检查列名。")
        
    return data

def TMEH_Optical_combine(ref_path: str,
                         src_path: str,
                         out_path: str):

    # ref_path = f'/media/minyukui/paper/Paper5_data/Input_data/JS/{region}_{year}_new9.tif'
    # src_path = f'/media/minyukui/paper/Paper5_data/TMEH/JS/{region}_TMEH_{year}.tif'
    # out_path = f'/media/minyukui/paper/Paper5_data/combine_FB/{region}_{year}_combine.tif'

    # =====================================================
    # 1. REF
    # =====================================================
    with rasterio.open(ref_path) as ref:
        ref_data = ref.read()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_profile = ref.profile.copy()
        ref_nodata = ref.nodata
        height, width = ref.height, ref.width

        ref_bounds = ref.bounds

    # =====================================================
    # 2. SRC
    # =====================================================
    with rasterio.open(src_path) as src:
        src_data = src.read()
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        src_bounds = src.bounds
        src_count = src.count

    # =====================================================
    # 3. 判断：CRS 是否“被错误标注”
    # =====================================================
    bounds_diff = (
        abs(ref_bounds.left   - src_bounds.left)   +
        abs(ref_bounds.right  - src_bounds.right)  +
        abs(ref_bounds.top    - src_bounds.top)    +
        abs(ref_bounds.bottom - src_bounds.bottom)
    )

    FORCE_ASSIGN_CRS = False

    if src_crs != ref_crs and bounds_diff < 1000:  # 1 km 阈值，极其保守
        print("[WARN] SRC CRS likely mislabeled (UTM zone mismatch)")
        print("[WARN] Forcing SRC CRS = REF CRS (NO reprojection)")
        FORCE_ASSIGN_CRS = True

    # =====================================================
    # 4. 对齐到 REF
    # =====================================================
    aligned_data = np.full(
        (src_count, height, width),
        ref_nodata if ref_nodata is not None else 0,
        dtype=src_data.dtype
    )

    for i in range(src_count):
        reproject(
            source=src_data[i],
            destination=aligned_data[i],
            src_transform=src_transform,
            src_crs=ref_crs if FORCE_ASSIGN_CRS else src_crs,
            src_nodata=src_nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=ref_nodata,
            resampling=Resampling.nearest
        )

    # =====================================================
    # 5. 自检
    # =====================================================
    nz = np.count_nonzero(aligned_data)
    print(f"[CHECK] SRC non-zero pixels: {nz}")

    if nz == 0:
        raise RuntimeError("SRC still empty after alignment — STOP")

    # =====================================================
    # 6. 合并 & 输出
    # =====================================================
    merged = np.concatenate([ref_data, aligned_data], axis=0)

    ref_profile.update(
        count=merged.shape[0],
        dtype=merged.dtype,
        transform=ref_transform,
        crs=ref_crs,
        nodata=ref_nodata
    )

    with rasterio.open(out_path, 'w', **ref_profile) as dst:
        dst.write(merged)

    # print(f"[DONE] {region} {year}: {merged.shape[0]} bands written")

def create_training_patches(
    csv_path: str, 
    tif_path: str, 
    patch_size: int ,
    output_dir: str #= '/home/minyukui/A_transform/Trainning_dataset'
):
    """
    
    根据 ICESat-2 点的地理坐标，从多波段 GeoTIFF 中提取指定大小的图像块。

    Args:
        csv_path (str): ICESat-2 点数据 (lat, lon, elevation) 的 CSV 文件路径。
        tif_path (str): 多波段光学影像 GeoTIFF 文件路径。
        patch_size (int): 图像块的边长 (例如 15 表示 15x15)。
        output_dir (str): 存储提取的图像块和标签的输出目录。

    """
    
    # --- 1. 数据准备和初始化 ---
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 ICESat-2 数据
    # 假设 CSV 文件包含 'lat', 'lon', 'elevation' 三列
    try:
        # df_icesat = pd.read_csv(csv_path)
        df_icesat = gpd.read_file(csv_path)

    except FileNotFoundError:

        print(f"错误: 找不到 CSV 文件 {csv_path}")
        return
    
    # 确定图像块的半边长
    half_patch = patch_size // 2
    
    # 用于存储所有样本的列表
    all_patches = []
    all_labels = []
    
    # --- 2. 打开 GeoTIFF 文件 ---
    print(f"正在打开 GeoTIFF 文件: {tif_path}")
    
    try:
        with rasterio.open(tif_path) as src:
            # 记录 CRS 信息（通常是投影坐标系）
            raster_crs = src.crs
            print(f"GeoTIFF CRS: {raster_crs}")
            
            transformer = Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
            # --- 3. 循环处理每个 ICESat-2 点 ---
            print(f"总共 {len(df_icesat)} 个 ICESat-2 点需要处理...")
            
            for index, row in tqdm(df_icesat.iterrows(), total=len(df_icesat), desc="提取图像块"):
                lon, lat, elevation = row['mean_lon'], row['mean_lat'], row['OH']
                x, y = transformer.transform(lon, lat)
                # --- 3.1 坐标转换（地理坐标到像素坐标）---
                # 使用 rasterio.index() 方法将地理坐标 (lon, lat) 转换为 (row, col)
                # 注意: rasterio 默认使用 WGS84/EPSG:4326 作为输入地理坐标
                
                try:
                    row_idx, col_idx = src.index(x, y)
                except Exception as e:
                    # 如果坐标点在 GeoTIFF 范围之外，跳过
                    # print(f"警告: 点 ({lon}, {lat}) 位于栅格范围外，已跳过。")
                    continue
                
                # --- 3.2 定义提取窗口（Window）---
                # 确定窗口的左上角 (col_off, row_off)
                col_off = col_idx - half_patch
                row_off = row_idx - half_patch
                
                # 创建一个 rasterio.windows.Window 对象
                window = Window(col_off, row_off, patch_size, patch_size)
                
                # --- 3.3 边界检查和切片提取 ---
                
                # 检查窗口是否完全在栅格范围内
                if (col_off >= 0 and row_off >= 0 and 
                    (col_off + patch_size) <= src.width and 
                    (row_off + patch_size) <= src.height):
                    
                    # 读取图像块
                    # read(..., window=...) 是从 GeoTIFF 文件中高效读取子集的关键
                    patch = src.read(window=window)
                    
                    # 调整维度顺序 (C, H, W) -> (H, W, C) 以便后续DL处理或可视化
                    # 如果您的 DL 框架 (如 PyTorch) 偏好 (C, H, W)，则无需此步
                    patch = np.transpose(patch, (1, 2, 0)) 
                    
                    # --- 3.4 存储样本 ---
                    all_patches.append(patch)
                    all_labels.append(elevation)

                else:
                    # 图像块位于边界，不完整，跳过（或用零填充，但跳过更安全）
                    # print(f"警告: 图像块 {index} 位于边界，跳过。")
                    pass

    except rasterio.RasterioIOError as e:
        print(f"错误: 无法读取 GeoTIFF 文件 {tif_path}。请检查路径和文件完整性。")
        print(e)
        return
    
    # --- 4. 结果整理与保存 ---
    
    # 将列表转换为 NumPy 数组
    X_train = np.array(all_patches)
    Y_train = np.array(all_labels)
    
    # 打印最终统计信息
    print("\n--- 提取完成 ---")
    print(f"成功提取的样本数量: {len(Y_train)}")
    print(f"输入特征 (X_train) 形状: {X_train.shape} (N, H, W, C)")
    print(f"标签 (Y_train) 形状: {Y_train.shape} (N, )")
    
    return X_train, Y_train


def ICESAT_Refined(ICESat_path, mask_apply, oh_mask, Freq_mask, output_to_new_shp ,display=True):

    a = gpd.read_file(ICESat_path)

    if mask_apply:
        
        mask = (a['OH'] > oh_mask[0]) & (a['b1_FB6'] < Freq_mask[0])
        for j in range(1 ,len(oh_mask)):
            mask = mask | ((a['OH'] > oh_mask[j]) & (a['b1_FB6'] < Freq_mask[j]))
        a = a[~mask]

    y_true = a['OH'].values
    y_pred = a['b1_FB6'].values

    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_true[idx], y_pred[idx], z[idx]

    if mask_apply & output_to_new_shp:
        p = Path(ICESat_path)
        new_p = p.with_name(p.stem + "_refined" + p.suffix)
        print(f'New files output to {new_p}')
        a.to_file(new_p)

    if display:

        fig, ax = plt.subplots(figsize=(6, 6))

        sc = ax.scatter(
            x, y,
            c=z,
            s=8,
            cmap='viridis',
            edgecolor='none'
        )

        plt.title(f'{region}_{year}', fontsize=16)

        limsx = [-2, 4]
        limsy = [0, 1]
        ax.plot(limsx, limsy, 'r--', linewidth=1)

        textstr = f'num = {len(a)}'
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,   # 使用轴坐标系
            fontsize=16,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plt.tight_layout()
        plt.show()
        
        

def draw_validation_plot2(
    refer_name,
    validation_point_path,
    retrieved_DTM_name,
    output_path,
    name,
    xy_lim_range,
    vmax,
    kde_sample_size=5000   # ⭐关键参数
):

    # ======================
    # 1. 读取 + 筛选
    # ======================
    gdf = gpd.read_file(validation_point_path)[[refer_name, retrieved_DTM_name]]

    z = gdf.dropna(subset=[refer_name, retrieved_DTM_name])
    z = z[
        (z[refer_name] != 0) &
        (z[retrieved_DTM_name] != 0) &
        (z[refer_name] < 6) &
        (z[retrieved_DTM_name] > -5)
    ]
    
    z = z[z!=-999]
    z = z[z!=6]
    z = z[z!=0]
    z = z.dropna()


    x = z[refer_name].values
    y = z[retrieved_DTM_name].values

    # ======================
    # 2. 指标
    # ======================
    diff = y - x
    R2 = r2_score(x, y)
    MAE = np.mean(np.abs(diff))
    RMSE = np.sqrt(np.mean(diff**2))
    num = len(x)

    print(f"R²={R2:.3f}, MAE={MAE:.3f}, RMSE={RMSE:.3f}, N={num}")

    # ======================
    # 3. KDE（只用子样本）
    # ======================
    N = len(x)

    if N > kde_sample_size:
        idx_sample = np.random.choice(N, kde_sample_size, replace=False)
    else:
        idx_sample = np.arange(N)

    xy_sample = np.vstack([x[idx_sample], y[idx_sample]])

    kde = gaussian_kde(xy_sample)

    # ⭐ 对所有点计算密度（关键技巧）
    xy_all = np.vstack([x, y])
    z_density = kde(xy_all)

    # ======================
    # 4. 排序（保证视觉效果）
    # ======================
    idx = z_density.argsort()
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z_density[idx]

    # ======================
    # 5. 绘图
    # ======================
    fig, ax = plt.subplots()

    sc = ax.scatter(
        x_sorted,
        y_sorted,
        c=z_sorted,
        s=5,
        cmap='Spectral_r',
        vmin=0,
        vmax=vmax
    )

    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=14)

    xmin, xmax = xy_lim_range[0], xy_lim_range[1]

    # 1:1线
    ax.plot([xmin, xmax], [xmin, xmax], '--', color='gray', linewidth=0.8)

    # 文本
    textstr = f"R² = {R2:.2f}\nMAE = {MAE:.2f}\nRMSE = {RMSE:.2f}\nN = {num}"
    ax.text(0.52, 0.05, textstr, transform=ax.transAxes, fontsize=20)
    ax.text(0.05, 0.95, name, transform=ax.transAxes, fontsize=20)

    # 坐标轴
    ax.set_xlabel('Airborne LiDAR DEM', fontsize=18)
    ax.set_ylabel('Retrieved DEM', fontsize=18)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    ticks = np.arange(xmin, xmax + xy_lim_range[2], xy_lim_range[2])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=16)

    plt.show()
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()

    print("绘图完成！")