program Project1;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  System.Math,
  System.Threading;

type
  TVector = array of Double;
  TMatrix = array of TVector;

  TLayerWeights = array of TMatrix;
  TLayerBiases = array of TVector;

  TNetworkWeights = array of TLayerWeights;
  TNetworkBiases = array of TLayerBiases;

  TLayerBuffers = array of TVector;

  TFitnessInfo = record
    Fitness: Double;
    Index: Integer;
  end;


const
  LayerSizes: array[0..2] of Integer = (2, 6, 1);
  LayerCount = 2;

var
  Weights: TNetworkWeights;
  NewWeights: TNetworkWeights;

  Biases: TNetworkBiases;
  NewBiases: TNetworkBiases;

  StartValues: TVector;
  GoalValues: TVector;

  Fitnesses: TVector;

  BestFitnesses: array of TFitnessInfo;

  TempFitnessInfo: TFitnessInfo;

  FinalActivations: TLayerBuffers;

  i, j, a: Integer;
  layerIndex, neuronIndex, inputIndex: Integer;

  parentAmount: Integer;
  childIndex: Integer;
  parentIndex: Integer;
  bestNetworkIndex: Integer;
  generation: Integer;

  startVal: Double;
  currentVal: Double;
  goalVal: Double;
  oldGoalVal: Double;
  endVal: Double;

  currentInput: Double;
  goalInput: Double;
  sum: Double;
  differenceVal: Double;
  bestDistance: Double;

  // -------------------------------------------------
  // EINSTELLUNGEN
  // -------------------------------------------------

  taskAmount: Integer = 18;
  taskSize: Integer = 1000;
  networkAmount: Integer = 300;
  generationAmount: Integer = 400;
  iterationAmount: Integer = 2000;
  rewardBalance: Double = 0.01;
  mutationSize: Double = 0.2;
  threshold: Double = 0.000005;

procedure InitOneNetwork(
  var NW: TLayerWeights;
  var NB: TLayerBiases);
var
  L: Integer;
  Neuron: Integer;
  InputIndex: Integer;
begin
  SetLength(NW, LayerCount);
  SetLength(NB, LayerCount);

  for L := 0 to LayerCount - 1 do
  begin
    SetLength(NW[L], LayerSizes[L + 1]);
    SetLength(NB[L], LayerSizes[L + 1]);

    for Neuron := 0 to LayerSizes[L + 1] - 1 do
    begin
      SetLength(NW[L][Neuron], LayerSizes[L]);

      NB[L][Neuron] := (Random * 2.0) - 1.0;

      for InputIndex := 0 to LayerSizes[L] - 1 do
      begin
        NW[L][Neuron][InputIndex] :=
          (Random * 2.0) - 1.0;
      end;
    end;
  end;
end;

procedure InitActivationBuffer(var A: TLayerBuffers);
var
  L: Integer;
begin
  SetLength(A, LayerCount + 1);

  for L := 0 to LayerCount do
  begin
    SetLength(A[L], LayerSizes[L]);
  end;
end;

procedure CopyNetworkParams(
  const SourceW: TLayerWeights;
  const SourceB: TLayerBiases;
  var DestW: TLayerWeights;
  var DestB: TLayerBiases);
var
  L: Integer;
  Neuron: Integer;
  InputIndex: Integer;
begin
  for L := 0 to LayerCount - 1 do
  begin
    for Neuron := 0 to LayerSizes[L + 1] - 1 do
    begin
      DestB[L][Neuron] := SourceB[L][Neuron];

      for InputIndex := 0 to LayerSizes[L] - 1 do
      begin
        DestW[L][Neuron][InputIndex] :=
          SourceW[L][Neuron][InputIndex];
      end;
    end;
  end;
end;

function ForwardNetwork(
  const NW: TLayerWeights;
  const NB: TLayerBiases;
  const CurrentInput: Double;
  const GoalInput: Double;
  var Activations: TLayerBuffers): Double;
var
  L: Integer;
  Neuron: Integer;
  InputIndex: Integer;
  S: Double;
begin
  Activations[0][0] := CurrentInput;
  Activations[0][1] := GoalInput;

  for L := 0 to LayerCount - 1 do
  begin
    for Neuron := 0 to LayerSizes[L + 1] - 1 do
    begin
      S := NB[L][Neuron];

      for InputIndex := 0 to LayerSizes[L] - 1 do
      begin
        S := S +
          NW[L][Neuron][InputIndex] *
          Activations[L][InputIndex];
      end;

      Activations[L + 1][Neuron] := Tanh(S);
    end;
  end;

  Result := Activations[LayerCount][0];
end;

procedure QuickSortFitness(
  var Items: array of TFitnessInfo;
  Left, Right: Integer);
var
  L: Integer;
  R: Integer;
  Pivot: Double;
  Temp: TFitnessInfo;
begin
  L := Left;
  R := Right;

  Pivot := Items[(Left + Right) div 2].Fitness;

  repeat
    while Items[L].Fitness < Pivot do
      Inc(L);

    while Items[R].Fitness > Pivot do
      Dec(R);

    if L <= R then
    begin
      Temp := Items[L];
      Items[L] := Items[R];
      Items[R] := Temp;

      Inc(L);
      Dec(R);
    end;
  until L > R;

  if Left < R then
    QuickSortFitness(Items, Left, R);

  if L < Right then
    QuickSortFitness(Items, L, Right);
end;

begin
  Randomize;


  SetLength(Weights, networkAmount);
  SetLength(Biases, networkAmount);

  SetLength(NewWeights, networkAmount);
  SetLength(NewBiases, networkAmount);

  SetLength(Fitnesses, networkAmount);
  SetLength(BestFitnesses, networkAmount);

  SetLength(StartValues, taskAmount);
  SetLength(GoalValues, taskAmount);

  for i := 0 to networkAmount - 1 do
  begin
    InitOneNetwork(Weights[i], Biases[i]);
    InitOneNetwork(NewWeights[i], NewBiases[i]);
  end;

  for a := 0 to taskAmount - 1 do
  begin
    StartValues[a] := Random(taskSize) + 1;
    GoalValues[a] := Random(taskSize) + 1;
  end;

  for generation := 0 to generationAmount - 1 do
  begin

    if generation > 0 then
    begin
      for i := 0 to networkAmount - 1 do
      begin
        CopyNetworkParams(
          NewWeights[i],
          NewBiases[i],
          Weights[i],
          Biases[i]
        );
      end;
    end;

    TParallel.For(0, networkAmount - 1,
      procedure(i: Integer)
      var
        a, j: Integer;

        localCurrentVal: Double;
        localGoalVal: Double;

        localBestDistance: Double;
        localDifferenceVal: Double;
        localEndVal: Double;

        localCurrentInput: Double;
        localGoalInput: Double;
        localSum: Double;

        localActivations: TLayerBuffers;

      begin
        Fitnesses[i] := 0;

        InitActivationBuffer(localActivations);

        for a := 0 to taskAmount - 1 do
        begin
          localCurrentVal := StartValues[a];
          localGoalVal := GoalValues[a];

          localBestDistance :=
            Abs(localGoalVal - localCurrentVal);

          for j := 1 to iterationAmount do
          begin

            localCurrentInput :=
              localCurrentVal / taskSize;

            localGoalInput :=
              localGoalVal / taskSize;

            localSum := ForwardNetwork(
              Weights[i],
              Biases[i],
              localCurrentInput,
              localGoalInput,
              localActivations
            );

            if localSum > threshold then
            begin
              localCurrentVal :=
                localCurrentVal + 1;
            end

            else if localSum < -threshold then
            begin
              localCurrentVal :=
                localCurrentVal - 1;
            end;

            localDifferenceVal :=
              Abs(localGoalVal - localCurrentVal);

            if localDifferenceVal < localBestDistance then
            begin
              localBestDistance := localDifferenceVal;
            end;
          end;

          localEndVal := localCurrentVal;

          Fitnesses[i] := Fitnesses[i] +
            (
              Abs(localGoalVal - localEndVal) +
              rewardBalance * localBestDistance
            );
        end;
      end
    );

    for i := 0 to networkAmount - 1 do
    begin
      BestFitnesses[i].Fitness := Fitnesses[i];
      BestFitnesses[i].Index := i;
    end;

    QuickSortFitness(
      BestFitnesses,
      0,
      networkAmount - 1
    );

    writeln(
      'Generation: ', generation,
      ' | Beste Fitness: ',
      BestFitnesses[0].Fitness:0:4
    );

    if generation = generationAmount - 1 then
      Continue;

    parentAmount := networkAmount div 10;

    if parentAmount < 1 then
      parentAmount := 1;

    if parentAmount > 10 then
      parentAmount := 10;

    if parentAmount > networkAmount then
      parentAmount := networkAmount;

    for i := 0 to parentAmount - 1 do
    begin
      parentIndex := BestFitnesses[i].Index;

      CopyNetworkParams(
        Weights[parentIndex],
        Biases[parentIndex],
        NewWeights[i],
        NewBiases[i]
      );
    end;

    for childIndex := parentAmount to networkAmount - 1 do
    begin

      parentIndex :=
        (childIndex - parentAmount) mod parentAmount;

      for layerIndex := 0 to LayerCount - 1 do
      begin

        for neuronIndex := 0 to LayerSizes[layerIndex + 1] - 1 do
        begin

          for inputIndex := 0 to LayerSizes[layerIndex] - 1 do
          begin
            NewWeights[childIndex][layerIndex][neuronIndex][inputIndex] :=
              NewWeights[parentIndex][layerIndex][neuronIndex][inputIndex]
              + (Random * (mutationSize * 2))
              - mutationSize;
          end;

          NewBiases[childIndex][layerIndex][neuronIndex] :=
            NewBiases[parentIndex][layerIndex][neuronIndex]
            + (Random * (mutationSize * 2))
            - mutationSize;
        end;
      end;
    end;
  end;

  bestNetworkIndex := BestFitnesses[0].Index;

  writeln;
  writeln('=======================================');
  writeln('Finaler Test des besten Netzwerkes');
  writeln('=======================================');

  writeln(
    'Trainings-Fitness: ',
    BestFitnesses[0].Fitness:0:4
  );

  InitActivationBuffer(FinalActivations);

  startVal := (Random(taskSize) + 1) * 100;
  currentVal := startVal;
  goalVal := (Random(taskSize) + 1) * 100;

  bestDistance := Abs(goalVal - currentVal);

  writeln('Startwert: ', startVal:0:0);
  writeln('Erstes Ziel: ', goalVal:0:0);

  for j := 1 to iterationAmount * 100 do
  begin

    currentInput := currentVal / (taskSize * 100);
    goalInput := goalVal / (taskSize * 100);

    sum := ForwardNetwork(
      Weights[bestNetworkIndex],
      Biases[bestNetworkIndex],
      currentInput,
      goalInput,
      FinalActivations
    );

    if sum > threshold then
    begin
      currentVal := currentVal + 1;
    end
    else if sum < -threshold then
    begin
      currentVal := currentVal - 1;
    end;

    differenceVal := Abs(goalVal - currentVal);

    if differenceVal < bestDistance then
      bestDistance := differenceVal;


    if j = iterationAmount div 2 then
    begin
      oldGoalVal := goalVal;

      repeat
        goalVal := (Random(taskSize) + 1) * 100;
      until goalVal <> oldGoalVal;

      bestDistance := Abs(goalVal - currentVal);

      writeln;
      writeln('*** ZIEL WIRD GEÄNDERT ***');
      writeln('Altes Ziel: ', oldGoalVal:0:0);
      writeln('Neues Ziel: ', goalVal:0:0);
    end;
  end;

  endVal := currentVal;

  writeln;
  writeln('=======================================');
  writeln('Finales Ergebnis');
  writeln('=======================================');

  writeln('Start: ', startVal:0:0);
  writeln('Letztes Ziel: ', goalVal:0:0);
  writeln('Endwert: ', endVal:0:0);
  writeln(
    'Abstand zum letzten Ziel: ',
    Abs(goalVal - endVal):0:2
  );

  writeln(
    'Fitness fuer das letzte Ziel: ',
    (
      Abs(goalVal - endVal) +
      rewardBalance * bestDistance
    ):0:4
  );

  writeln;
  writeln('Taste druecken zum Beenden.');
  readln;
end.
