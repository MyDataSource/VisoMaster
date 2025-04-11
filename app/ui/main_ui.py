from typing import Dict
from pathlib import Path
from functools import partial
import copy
import logging

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import GraphicsViewEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter, ListWidgetEventFilter
from app.ui.widgets import ui_workers
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.helpers.typing_helper import FacesParametersTypes, ParametersTypes, ControlTypes, MarkerTypes

ParametersWidgetTypes = Dict[str, widget_components.ToggleButton|widget_components.SelectionBox|widget_components.ParameterDecimalSlider|widget_components.ParameterSlider|widget_components.ParameterText]

logger = logging.getLogger(__name__)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)
    def initialize_variables(self):
        logger.debug("开始初始化变量...")
        try:
            self.video_loader_worker: ui_workers.TargetMediaLoaderWorker|bool = False
            self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker|bool = False
            self.target_videos_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='target_videos')
            self.input_faces_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='input_faces')
            self.merged_embeddings_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
            self.video_processor = VideoProcessor(self)
            self.models_processor = ModelsProcessor(self)
            logger.debug("已初始化处理器")
            
            self.target_videos: Dict[int, widget_components.TargetMediaCardButton] = {}
            self.target_faces: Dict[int, widget_components.TargetFaceCardButton] = {}
            self.input_faces: Dict[int, widget_components.InputFaceCardButton] = {}
            self.merged_embeddings: Dict[int, widget_components.EmbeddingCardButton] = {}
            logger.debug("已初始化媒体字典")
            
            self.cur_selected_target_face_button: widget_components.TargetFaceCardButton = False
            self.selected_video_button: widget_components.TargetMediaCardButton = False
            self.selected_target_face_id = False
            
            self.parameters: FacesParametersTypes = {}
            self.default_parameters: ParametersTypes = {}
            self.copied_parameters: ParametersTypes = {}
            self.current_widget_parameters: ParametersTypes = {}
            logger.debug("已初始化参数")
            
            self.markers: MarkerTypes = {}
            self.parameters_list = {}
            self.control: ControlTypes = {}
            self.parameter_widgets: ParametersWidgetTypes = {}
            self.loaded_embedding_filename: str = ''
            
            self.last_target_media_folder_path = ''
            self.last_input_media_folder_path = ''
            
            self.is_full_screen = False
            self.dfm_models_data = DFM_MODELS_DATA
            self.loading_new_media = False
            logger.debug("已初始化其他变量")
            
            self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
            self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))
            self.model_loading_signal.connect(partial(common_widget_actions.show_model_loading_dialog, self))
            self.model_loaded_signal.connect(partial(common_widget_actions.hide_model_loading_dialog, self))
            self.display_messagebox_signal.connect(partial(common_widget_actions.create_and_show_messagebox, self))
            logger.debug("已连接信号")
            
        except Exception as e:
            logger.error(f"初始化变量时出错: {str(e)}", exc_info=True)
            raise
    def initialize_widgets(self):
        logger.debug("开始初始化组件...")
        try:
            self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
            self.targetVideosList.setWrapping(True)
            self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

            self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
            self.inputFacesList.setWrapping(True)
            self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

            layout_actions.set_up_menu_actions(self)

            list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
            list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

            self.targetVideosList.setAcceptDrops(True)
            self.targetVideosList.viewport().setAcceptDrops(False)
            self.inputFacesList.setAcceptDrops(True)
            self.inputFacesList.viewport().setAcceptDrops(False)
            list_widget_event_filter = ListWidgetEventFilter(self, self)
            self.targetVideosList.installEventFilter(list_widget_event_filter)
            self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
            self.inputFacesList.installEventFilter(list_widget_event_filter)
            self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)

            self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
            self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))

            self.scene = QtWidgets.QGraphicsScene()
            self.graphicsViewFrame.setScene(self.scene)
            graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
            self.graphicsViewFrame.installEventFilter(graphics_event_filter)

            video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

            video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
            self.videoSeekSlider.installEventFilter(video_slider_event_filter)
            self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
            self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
            self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
            video_control_actions.set_up_video_seek_slider(self)
            self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
            self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))

            self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
            self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
            self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
            self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))

            self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
            video_control_actions.set_up_video_seek_line_edit(self)
            video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
            self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

            self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
            self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
            self.findTargetFacesButton.clicked.connect(partial(card_actions.find_target_faces, self))
            self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
            self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filter_target_videos, self))
            self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
            self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
            self.filterWebcamsCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
            self.filterWebcamsCheckBox.clicked.connect(partial(list_view_actions.load_target_webcams, self))

            self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filter_input_faces, self))
            self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filter_merged_embeddings, self))
            self.openEmbeddingButton.clicked.connect(partial(save_load_actions.open_embeddings_from_file, self))
            self.saveEmbeddingButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self))
            self.saveEmbeddingAsButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self, True))

            self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
            self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))

            self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))
            self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))

            self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
            self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
            self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))

            self.faceMaskCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
            self.faceCompareCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))

            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=COMMON_LAYOUT_DATA, layoutWidget=self.commonWidgetsLayout, data_type='parameter')
            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

            self.outputFolderButton.clicked.connect(partial(list_view_actions.select_output_media_folder, self))
            common_widget_actions.create_control(self, 'OutputMediaFolder', '')

            self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)

            video_control_actions.reset_media_buttons(self)

            font = self.vramProgressBar.font()
            font.setBold(True)
            self.vramProgressBar.setFont(font)
            common_widget_actions.update_gpu_memory_progressbar(self)
            self.tabWidget.setCurrentIndex(0)
        except Exception as e:
            logger.error(f"初始化组件时出错: {str(e)}", exc_info=True)
            raise
    def __init__(self):
        logger.info("正在创建主窗口...")
        try:
            super(MainWindow, self).__init__()
            self.setupUi(self)
            self.initialize_variables()
            self.initialize_widgets()
            self.load_last_workspace()
            logger.info("主窗口创建完成")
        except Exception as e:
            logger.error(f"创建主窗口时出错: {str(e)}", exc_info=True)
            raise

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if self.scene.items():
            pixmap_item = self.scene.items()[0]
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect )

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_F11:
                video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_V:
                video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D:
                video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_A:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_Z:
                self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key_Space:
                self.buttonMediaPlay.click()
            case QtCore.Qt.Key_R:
                self.buttonMediaRecord.click()
            case QtCore.Qt.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    video_control_actions.remove_video_slider_marker(self)
                else:
                    video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key_W:
                video_control_actions.move_slider_to_nearest_marker(self, 'next')
            case QtCore.Qt.Key_Q:
                video_control_actions.move_slider_to_nearest_marker(self, 'previous')
            case QtCore.Qt.Key_S:
                self.swapfacesButton.click()

    def closeEvent(self, event):
        logger.info("正在关闭应用程序...")
        try:
            self.video_processor.stop_processing()
            list_view_actions.clear_stop_loading_input_media(self)
            list_view_actions.clear_stop_loading_target_media(self)
            save_load_actions.save_current_workspace(self, 'last_workspace.json')
            logger.info("应用程序已安全关闭")
            event.accept()
        except Exception as e:
            logger.error(f"关闭应用程序时出错: {str(e)}", exc_info=True)
            event.accept()

    def load_last_workspace(self):
        if Path('last_workspace.json').is_file():
            load_dialog = widget_components.LoadLastWorkspaceDialog(self)
            load_dialog.exec_()

    def save_last_workspace(self):
        pass