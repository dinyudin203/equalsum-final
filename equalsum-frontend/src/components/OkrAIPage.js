import React, { useState, useEffect, useCallback} from 'react';
import { getUniqueAIData, getTaskStatus } from '../api/api';
import * as XLSX from 'xlsx';

const OkrAIPage = ({ aiOkrId = [] }) => {
  const [currentDataList, setCurrentDataList] = useState([]); // All fetched data
  const [currentIndex, setCurrentIndex] = useState(0); // Track current page
  const [completedTasks, setCompletedTasks] = useState([]); // Track completed task_ids
  const [pendingTasks, setPendingTasks] = useState([]); // Track pending task_ids
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isPending_start, setIsPending_start] = useState(true);

  const currentDate = new Date().toISOString().slice(0, 10).replace(/-/g, '');


  // Debugging aiOkrId
  useEffect(() => {
    console.log('aiOkrId:', aiOkrId, 'Type:', typeof aiOkrId);
  }, [aiOkrId]);

  // Fetch data for all items
  const fetchDataForId = useCallback(async (item) => {
    try {
      const response = await getUniqueAIData(item.id);
      setCurrentDataList((prev) => {
        const updatedList = prev.filter((data) => data.task_id !== item.task_id); // 기존 중복 제거
        return [...updatedList, { ...response.data, task_id: item.task_id }];
      });
      setPendingTasks((prev) => prev.filter((taskId) => taskId !== item.task_id));
    } catch (err) {
      console.error(`Failed to fetch data for id ${item.id}:`, err);
    }
  }, []); 

  useEffect(() => {
    if (!Array.isArray(aiOkrId)) {
      console.error('Invalid aiOkrId. Expected an array but got:', aiOkrId);
      return;
    }
    aiOkrId.forEach((item) => {
      fetchDataForId(item);
    });

  }, [aiOkrId, fetchDataForId]);

  useEffect(() => {
    if (Array.isArray(aiOkrId)) {
      const allPendingTasks = aiOkrId.map((item) => item.task_id);
      setPendingTasks(allPendingTasks);
    }
  }, [aiOkrId]);
  

  // Check task statuses periodically
  const checkTaskStatus = useCallback(async () => {
    if (!Array.isArray(aiOkrId)) return;
  
    const statusPromises = aiOkrId.map(async (item) => {
      // PENDING 상태이거나 아직 완료되지 않은 항목만 상태 확인
      if (!completedTasks.includes(item.task_id)) {
        const response = await getTaskStatus(item.task_id);
        const response_state = String(response.data.task_status);
        const checking_true = (response_state === 'PENDING')
        console.log('State', item.id, ':', response_state, checking_true);
  
        if (response_state === 'SUCCESS') {
          setCompletedTasks((prev) => [...prev, item.task_id]);
          setPendingTasks((prev) => prev.filter((taskId) => taskId !== item.task_id)); // PENDING에서 제거
          await fetchDataForId(item); // 성공 시 데이터 가져오기
          if(isPending_start)
            setIsPending_start(false);
        } else if (response_state === 'PENDING') {
          // PENDING 상태를 유지
          setPendingTasks((prev) => {
            if (!prev.includes(item.task_id)) {
              console.log('PENDING 상태 추가:', item.task_id);
              return [...prev, item.task_id];
            }
            return prev;
          });

          console.log('isPending 상태 확인:', isPending);
          if(isPending_start)
            setIsPending_start(false);
                
        } else {
          console.warn(`Task ${item.task_id} returned unexpected status: ${response_state}`);
        }
      }
    });
  
    await Promise.all(statusPromises); // 모든 상태 체크가 끝날 때까지 기다림
  }, [aiOkrId, completedTasks, pendingTasks, fetchDataForId]);    

  useEffect(() => {
    const intervalId = setInterval(() => {
      checkTaskStatus();
    }, 5000);
    return () => clearInterval(intervalId);
  }, [checkTaskStatus]);

  // Handle exporting all data
  const handleExport = () => {
    if (!currentDataList.length) {
      alert('No data available for export!');
      return;
    }
    // PENDING 상태가 아닌 데이터만 필터링
    const filteredData = currentDataList.filter(
      (data) => !pendingTasks.includes(data.task_id)
    );

    if (filteredData.length === 0) {
      alert('No completed data available for export!');
      return;
    }

    const uniqueData = Array.from(
      new Map(filteredData.map((item) => [item.task_id, item])).values()
    );
  
    if (uniqueData.length === 0) {
      alert('No completed data available for export!');
      return;
    }

    const exportData = currentDataList.map((data) => {
      const aiOkrItem = aiOkrId.find((item) => item.task_id === data.task_id);
      const baseExportData = {
        'No': data.okr_id,
        '일자': aiOkrItem?.date || '',
        '기업명': aiOkrItem?.companyName || '',
        '부서명': aiOkrItem?.department || '',
        '구분': aiOkrItem?.type || '',
        '상위/해당목표': data.upper_objective || '',
        '작성 OKR': data.input_sentence || '',
        '수정 OKR': data.revision || '',
        '수정이유': data.revision_description || '',
        '가이드라인': data.guideline || '',
      };

      if (aiOkrItem?.type === 'Key Result') {
        baseExportData['연관성_점수'] = 'N/A';
        baseExportData['연관성_이유'] = 'N/A';
        baseExportData['고객가치_점수'] = 'N/A';
        baseExportData['고객가치_이유'] = 'N/A';
        baseExportData['연결성_점수'] = data.predictions?.[0]?.prediction_score || 'N/A';
        baseExportData['연결성_이유'] = data.predictions?.[0]?.prediction_description || 'N/A';
        baseExportData['측정가능성_점수'] = data.predictions?.[1]?.prediction_score || 'N/A';
        baseExportData['측정가능성_이유'] = data.predictions?.[1]?.prediction_description || 'N/A';
        baseExportData['렬과지향성_점수'] = data.predictions?.[2]?.prediction_score || 'N/A';
        baseExportData['결과지향성_이유'] = data.predictions?.[2]?.prediction_description || 'N/A';
      } else if (aiOkrItem?.type === 'Objective') {
        baseExportData['연관성_점수'] = data.predictions?.[0]?.prediction_score || 'N/A';
        baseExportData['연관성_이유'] = data.predictions?.[0]?.prediction_description || 'N/A';
        baseExportData['고객가치_점수'] = data.predictions?.[1]?.prediction_score || 'N/A';
        baseExportData['고객가치_이유'] = data.predictions?.[1]?.prediction_description || 'N/A';
        baseExportData['연결성_점수'] = 'N/A';
        baseExportData['연결성_이유'] = 'N/A';
        baseExportData['측정가능성_점수'] = 'N/A';
        baseExportData['측정가능성_이유'] = 'N/A';
        baseExportData['렬과지향성_점수'] = 'N/A';
        baseExportData['결과지향성_이유'] = 'N/A';
      }

      return baseExportData;
    });

    const worksheet = XLSX.utils.json_to_sheet(exportData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'OKR Data');
    XLSX.writeFile(workbook, `OKR_Data_${currentDate}.xlsx`);
  };

  // Navigate to the next page
  const handleNext = () => {
    if (currentIndex < aiOkrId.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  // Navigate to the previous page
  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  };

  // Get the current data for the selected index
  const currentAiOkr = aiOkrId[currentIndex];
  const currentData = currentDataList.find((data) => data.task_id === currentAiOkr?.task_id);
  // const isPending = pendingTasks.includes(currentAiOkr?.task_id);
  // const isPending = useMemo(() => pendingTasks.includes(currentAiOkr?.task_id), [pendingTasks, currentAiOkr]);
  const isPending = pendingTasks.includes(currentAiOkr?.task_id);

  function FormattedText({ text }) {
    return <div style={{ whiteSpace: 'pre-line', lineHeight: '1.5' }}>{text}</div>;
  }

  return (
    <div className="page-container">
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '20px', // Optional spacing below the header
        }}> 
        <h1>AI 적용 결과 페이지</h1>
        <button
          onClick={handleExport}
          style={{
            padding: '10px 15px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        >
          Export
        </button>
      </div> 
      {error && <h2 style={{ color: 'red' }}>{error}</h2>}
      { loading ? (
        <h2 style={{ textAlign: 'center', marginTop: '200px', marginBottom: '300px' }}>Loading...</h2>
        ) : (isPending_start || isPending)  && currentData ? (
          <h2 style={{ textAlign: 'center', marginTop: '200px' , marginBottom: '300px'}}>
            Task is still pending. 
            <h3></h3>
            Please wait.
          </h2>
        ) : currentData ? (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px' }}>
          <h3>OKR {currentIndex + 1}</h3>
          <p><strong>기업명:</strong> {currentAiOkr?.companyName}</p>
          <p><strong>부서명:</strong> {currentAiOkr?.department}</p>
          <p><strong>구분(OKR):</strong> {currentAiOkr?.type}</p>
          <p><strong>상위/해당목표:</strong> {currentData.upper_objective}</p>
          <p><strong>작성 OKR:</strong> {currentData.input_sentence}</p>
 
          <h3>Revision</h3>
          <p><strong>수정된 OKR:</strong> {currentData.revision}</p>
          <p><strong>수정한 이유:</strong> {currentData.revision_description}</p>

          <h3>Guideline</h3>
          <p><FormattedText text={currentData.guideline} /></p>
          {currentData.predictions && currentData.predictions.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h3>Evaluation</h3>
              {currentData.predictions.map((prediction, index) => (
                <div
                  key={index}
                  style={{
                    borderBottom: '1px solid #ccc',
                    marginBottom: '10px',
                    paddingBottom: '10px',
                  }}
                >
                  <p>
                    <strong>평가 기준:</strong> {prediction.prediction_type || 'N/A'}
                  </p>
                  <p>
                    <strong>점수:</strong> {prediction.prediction_score || 'N/A'}
                  </p>
                  <p>
                    <strong>날짜:</strong> {prediction.prediction_date
                      ? prediction.prediction_date.split('T')[0]
                      : 'N/A'}
                  </p>
                  <p>
                    <strong>평가 이유:</strong>
                    <FormattedText text={prediction.prediction_description || 'N/A'} />
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>):(
            <h2 style={{ marginTop: '230px', marginBottom: '300px',textAlign: 'center' }}>
              No data available.
            </h2>
          )
      }

      {/* Pagination Buttons */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: '20px' }}>
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          style={{
            padding: '5px 10px',
            backgroundColor: currentIndex === 0 ? '#ccc' : '#007bff',
            color: 'white',
            cursor: currentIndex === 0 ? 'not-allowed' : 'pointer',
          }}
        >
          이전
        </button>
        <span>
          {aiOkrId.length > 0 ? `${currentIndex + 1} / ${aiOkrId.length}` : '0 / 0'}
        </span>
        <button
          onClick={handleNext}
          disabled={currentIndex === aiOkrId.length - 1}
          style={{
            padding: '5px 10px',
            backgroundColor: ( currentIndex === aiOkrId.length - 1 || aiOkrId.length == 0)  ? '#ccc' : '#007bff',
            color: 'white',
            cursor: currentIndex === aiOkrId.length - 1 ? 'not-allowed' : 'pointer',
          }}
        >
          다음
        </button>
      </div>
    </div>
  );
};

export default OkrAIPage;
