import logging
import sys
import platform
print('import torch')
import torch
print('import app.ui')
from app.ui import main_ui
print('import PySide6')
from PySide6 import QtWidgets 
print('import qdarktheme')
import qdarktheme
print('import app.ui.core.proxy_style')
from app.ui.core.proxy_style import ProxyStyle

# 打印 Python 版本信息
print(f"Python 版本: {platform.python_version()}")
print(f"Python 实现: {platform.python_implementation()}")
print(f"Python 编译器: {platform.python_compiler()}")
print(f"操作系统: {platform.system()} {platform.release()}")

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visomaster.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# MPS 测试
logger.info("正在检查 MPS 支持...")
logger.info(f"PyTorch 版本: {torch.__version__}")
logger.info(f"Python 版本: {platform.python_version()}")
logger.info(f"MPS 可用: {torch.backends.mps.is_available()}")
logger.info(f"MPS 已构建: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # 测试基本操作
    x = torch.ones(5, device=device)
    y = torch.ones(5, device=device)
    z = x + y
    logger.info(f"基本操作测试: {z}")
    
    # 测试矩阵运算
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 2, device=device)
    c = torch.matmul(a, b)
    logger.info(f"矩阵运算测试: {c}")
    
    logger.info("MPS 工作正常！")
else:
    logger.warning("MPS 不可用")

if __name__=="__main__":
    try:
        logger.info("正在启动 VisoMaster...")
        
        app = QtWidgets.QApplication(sys.argv)
        logger.debug("QApplication 已创建")
        
        app.setStyle(ProxyStyle())
        logger.debug("已设置代理样式")
        
        try:
            with open("app/ui/styles/dark_styles.qss", "r") as f:
                _style = f.read()
                _style = qdarktheme.load_stylesheet(custom_colors={"primary": "#4facc9"})+'\n'+_style
                app.setStyleSheet(_style)
                logger.debug("已加载并应用样式表")
        except Exception as e:
            logger.error(f"加载样式表时出错: {str(e)}")
        
        logger.info("正在初始化主窗口...")
        window = main_ui.MainWindow()
        window.show()
        logger.info("主窗口已显示，开始运行应用程序...")
        
        app.exec()
    except Exception as e:
        logger.critical(f"应用程序启动失败: {str(e)}", exc_info=True)
        sys.exit(1)