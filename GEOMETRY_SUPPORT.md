# Bubblify 多几何形状支持

## 概述

Bubblify 现在支持多种几何形状类型，不仅仅是球形(sphere)，还包括盒子(box)和圆柱体(cylinder)。这为机器人碰撞检测提供了更精确和灵活的几何近似。

## 支持的几何形状

### 1. 球形 (Sphere)
- **参数**: 半径 (radius)
- **用途**: 适用于关节、圆形部件的近似
- **URDF表示**: `<sphere radius="0.05"/>`

### 2. 盒子 (Box)
- **参数**: 长度、宽度、高度 (length, width, height)
- **用途**: 适用于矩形部件、连杆的近似
- **URDF表示**: `<box size="0.1 0.1 0.1"/>`

### 3. 圆柱体 (Cylinder)
- **参数**: 半径 (radius)、高度 (height)
- **用途**: 适用于圆形连杆、轴的近似
- **URDF表示**: `<cylinder radius="0.05" length="0.1"/>`

## GUI 使用说明

### 几何形状选择
1. 在"🔶 Geometry Editor"面板中选择要添加的几何形状类型
2. 使用"Geometry Type"下拉菜单选择：sphere、box 或 cylinder

### 参数调整
根据选择的几何形状类型，相应的参数面板会显示：

- **⚪ Sphere Properties**: 调整球形半径
- **📦 Box Properties**: 调整盒子的长、宽、高
- **🥫 Cylinder Properties**: 调整圆柱体的半径和高度

### 操作流程
1. 选择目标连杆 (Link)
2. 选择几何形状类型 (Geometry Type)
3. 调整几何形状参数
4. 点击"➕ Add Geometry"添加几何形状
5. 使用3D变换控制器调整位置和旋转
6. 使用"🔄 Rotation Properties"面板精确调整RPY角度
7. 对于盒子几何形状，使用交互式调节轴直接调整尺寸
8. 重复以上步骤添加更多几何形状

## 交互式尺寸调节 🆕

### 盒子调节轴 (Box Resize Gizmos)
盒子几何形状提供了直观的3D尺寸调节功能：

- **6个调节轴**: 每个盒子在±X、±Y、±Z方向各有一个调节轴
- **表面定位**: 调节轴位于盒子表面，便于识别和操作
- **颜色编码**: 
  - 🔴 红色: X轴方向 (长度)
  - 🟢 绿色: Y轴方向 (宽度)  
  - 🔵 蓝色: Z轴方向 (高度)
- **实时更新**: 拖拽调节轴时，盒子尺寸和UI滑块同步更新
- **最小尺寸限制**: 防止盒子尺寸小于1cm，确保有效的碰撞检测
- **智能同步**: 调节一个轴时，其他轴自动调整位置以保持在盒子表面

### 使用方法
1. 创建或选择一个盒子几何形状
2. 在3D视图中会看到6个小的调节轴出现在盒子表面
3. 拖拽任意调节轴来改变对应方向的尺寸
4. 观察UI面板中的参数滑块同步更新
5. 盒子的3D可视化实时反映尺寸变化

## 旋转功能 🆕

### 3D旋转控制
- **3D变换控制器**: 在3D视图中直接拖拽旋转环来旋转几何形状（仅对盒子和圆柱体有效）
- **RPY滑块**: 在GUI面板中精确设置Roll、Pitch、Yaw角度（-π到π弧度）
- **实时同步**: 3D控制器和RPY滑块会实时同步显示当前旋转状态
- **球体优化**: 球体的旋转功能被禁用，因为球体是完全中心对称的

### 旋转表示
- **RPY角度**: Roll-Pitch-Yaw欧拉角（弧度）
- **四元数**: 内部使用四元数进行精确的旋转计算
- **自动转换**: RPY和四元数之间自动转换，保证精度

## 导出功能

### YAML 导出
导出的YAML文件包含所有几何形状类型和旋转信息：
```yaml
collision_geometries:
  base_link:
    - center: [0.0, 0.0, 0.0]
      type: sphere
      radius: 0.05
      # 注意：球体不包含旋转信息（因为是中心对称的）
    - center: [0.1, 0.0, 0.0]
      type: box
      size: [0.1, 0.1, 0.1]
      rpy: [0.785, 0.0, 0.0]  # 45度绕X轴旋转
    - center: [0.2, 0.0, 0.0]
      type: cylinder
      radius: 0.03
      height: 0.15
      rpy: [0.0, 0.0, 1.047]  # 60度绕Z轴旋转
```

### URDF 导出
导出的URDF文件会为每种几何形状类型生成相应的碰撞元素，包含旋转信息：
```xml
<collision name="sphere_0">
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
  <geometry>
    <sphere radius="0.05"/>
  </geometry>
</collision>
<collision name="box_1">
  <origin xyz="0.1 0.0 0.0" rpy="0.785 0.0 0.0"/>  <!-- 45度绕X轴旋转 -->
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
<collision name="cylinder_2">
  <origin xyz="0.2 0.0 0.0" rpy="0.0 0.0 1.047"/>  <!-- 60度绕Z轴旋转 -->
  <geometry>
    <cylinder radius="0.03" length="0.15"/>
  </geometry>
</collision>
```

## 向后兼容性

- 所有原有的球形相关功能都保持兼容
- 现有的YAML配置文件可以正常加载
- API保持向后兼容，`Sphere`类现在是`Geometry`类的别名

## 使用示例

```bash
# 启动Bubblify GUI
python -m bubblify.cli --urdf_path /path/to/robot.urdf

# 或使用现有的球形配置文件
python -m bubblify.cli --urdf_path /path/to/robot.urdf --spherization_yml existing_spheres.yml
```

## 技术细节

### 有效半径计算
- **球形**: 直接使用半径值
- **盒子**: 使用对角线长度的一半作为有效半径
- **圆柱体**: 使用圆柱体半径作为有效半径

### 数据结构
新的`Geometry`类包含所有几何形状类型的参数，根据`geometry_type`字段确定使用哪些参数。

这种设计确保了代码的简洁性和扩展性，同时保持了与现有代码的完全兼容性。
