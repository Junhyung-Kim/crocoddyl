///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   boost::shared_ptr<ActivationModelAbstract> activation,
                                                   boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, activation, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::~CostModelResidualTpl() {}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const VectorXs>& x) {
  // Compute the cost residual
  residual_->calc(data->residual, x);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact wrench cone residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);
  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  const bool is_rx = residual_->get_x_dependent();
  const bool is_rzmp = residual_->get_zmp_dependent();
  const bool is_ru = residual_->get_u_dependent() && nu_ != 0;
  const std::size_t nv = state_->get_nv();

  if (is_ru) {
    data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
    d->Arr_Ru.noalias() = data->activation->Arr.diagonal().asDiagonal() * data->residual->Ru;
    data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
  }
  if (is_rq && is_rv && is_rx &&(is_rzmp == false)) {
    data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.noalias() = data->activation->Arr.diagonal().asDiagonal() * data->residual->Rx;
    data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
    if (is_ru) {
      data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
    }
  } else if (is_rq && is_rv && (is_rx == false)&&(is_rzmp == false)) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(2*nv);
    data->Lx.head(2*nv).noalias() = Rq.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(2*nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    data->Lxx.topLeftCorner(2*nv, 2*nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(2*nv);
    if (is_ru) {
      data->Lxu.topRows(2*nv).noalias() = Rq.transpose() * d->Arr_Ru;
    }
  }
  else if (is_rq && (is_rv == false) && is_rx&&(is_rzmp == false)) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rx = data->residual->Rx.rightCols(8);
   // std::cout << "nr ss " << data->activation->Ar.size() << std::endl;
    data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
    data->Lx.tail(8).noalias() = Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    d->Arr_Rx.rightCols(8).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rx;
    data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(nv);
    data->Lxx.bottomRightCorner(8, 8).noalias() = Rx.transpose() * d->Arr_Rx.rightCols(8);
  } else if (is_rq) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
    data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(nv);
    if (is_ru) {
      data->Lxu.topRows(nv).noalias() = Rq.transpose() * d->Arr_Ru;
    }
  } else if (is_rv) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv = data->residual->Rx.middleCols(nv,nv);
    data->Lx.segment(nv,nv).noalias() = Rv.transpose() * data->activation->Ar;
    d->Arr_Rx.middleCols(nv,nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rv;
    data->Lxx.block(nv,nv,nv,nv).noalias() = Rv.transpose() * d->Arr_Rx.middleCols(nv,nv);
    if (is_ru) {
      data->Lxu.middleRows(nv, nv).noalias() = Rv.transpose() * d->Arr_Ru;
    }
  } else if (is_rx) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rx = data->residual->Rx.rightCols(8);
    data->Lx.tail(8).noalias() = Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.rightCols(8).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rx;
    data->Lxx.bottomRightCorner(8, 8).noalias() = Rx.transpose() * d->Arr_Rx.rightCols(8);
    if (is_ru) {
      data->Lxu.bottomRows(8).noalias() = Rx.transpose() * d->Arr_Ru;
    }
  }
  else if (is_rzmp) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rzmpx = data->residual->Rx.rightCols(6);//.leftCols(1);
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rzmpy = data->residual->Rx.rightCols(2);//.leftCols(1);
    data->Lx.tail(6).head(1).noalias() = Rzmpx.leftCols(1).transpose() * data->activation->Ar;
    data->Lx.tail(2).head(1).noalias() = Rzmpy.leftCols(1).transpose() * data->activation->Ar;
    d->Arr_Rx.rightCols(6).leftCols(1).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rzmpx.leftCols(1);
    d->Arr_Rx.rightCols(2).leftCols(1).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rzmpy.leftCols(1);   
    data->Lxx.bottomRightCorner(6, 6).topLeftCorner(1, 1).noalias() = Rzmpx.leftCols(1).transpose() * d->Arr_Rx.rightCols(6).leftCols(1);
    data->Lxx.bottomRightCorner(2, 2).topLeftCorner(1, 1).noalias() = Rzmpy.leftCols(1).transpose() * d->Arr_Rx.rightCols(2).leftCols(1);
  /*   if (is_ru) {
      data->Lxu.bottomRows(1).noalias() = Rzmp.leftCols(1).transpose() * d->Arr_Ru;
    }*/
  }
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x) {
  // Compute the derivatives of the activation and contact wrench cone residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  const bool is_rx = residual_->get_x_dependent();
  const bool is_rzmp = residual_->get_zmp_dependent();
  const std::size_t nv = state_->get_nv();
  if (is_rq && is_rv && is_rx &&(is_rzmp == false)) {
    data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.noalias() = data->activation->Arr.diagonal().asDiagonal() * data->residual->Rx;
    data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
  } else if (is_rq && is_rv && (is_rx == false)&&(is_rzmp == false)) {
    
Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(2*nv);
    data->Lx.head(2*nv).noalias() = Rq.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(2*nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    data->Lxx.topLeftCorner(2*nv, 2*nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(2*nv);
  } else if (is_rq && (is_rv == false) && is_rx&&(is_rzmp == false)) {   
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rx = data->residual->Rx.rightCols(8);
   // std::cout << "nr ss " << data->activation->Ar.size() << std::endl;
    data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
    data->Lx.tail(8).noalias() = Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    d->Arr_Rx.rightCols(8).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rx;
    data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(nv);
    data->Lxx.bottomRightCorner(8, 8).noalias() = Rx.transpose() * d->Arr_Rx.rightCols(8);
  }  else if (is_rq) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
    data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
    d->Arr_Rx.leftCols(nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rq;
    data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * d->Arr_Rx.leftCols(nv);
  } else if (is_rv) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv = data->residual->Rx.middleCols(nv,nv);
    data->Lx.segment(nv,nv).noalias() = Rv.transpose() * data->activation->Ar;
    d->Arr_Rx.middleCols(nv,nv).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rv;
    data->Lxx.block(nv,nv,nv,nv).noalias() = Rv.transpose() * d->Arr_Rx.middleCols(nv, nv);
  } else if (is_rx) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv = data->residual->Rx.rightCols(8);
    data->Lx.tail(8).noalias() = Rv.transpose() * data->activation->Ar;
    d->Arr_Rx.rightCols(8).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rv;
    data->Lxx.bottomRightCorner(8, 8).noalias() = Rv.transpose() * d->Arr_Rx.rightCols(8);
  } else if (is_rzmp) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rzmpx = data->residual->Rx.rightCols(6);
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rzmpy = data->residual->Rx.rightCols(2);
    data->Lx.tail(6).head(1).noalias() = Rzmpx.leftCols(1).transpose() * data->activation->Ar;
    data->Lx.tail(2).head(1).noalias() = Rzmpy.leftCols(1).transpose() * data->activation->Ar;
    d->Arr_Rx.rightCols(6).leftCols(1).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rzmpx.leftCols(1);
    d->Arr_Rx.rightCols(2).leftCols(1).noalias() = data->activation->Arr.diagonal().asDiagonal() * Rzmpy.leftCols(1);
    data->Lxx.bottomRightCorner(6, 6).topLeftCorner(1, 1).noalias() = Rzmpx.leftCols(1).transpose() * d->Arr_Rx.rightCols(6).leftCols(1);
    data->Lxx.bottomRightCorner(2, 2).topLeftCorner(1, 1).noalias() = Rzmpy.leftCols(1).transpose() * d->Arr_Rx.rightCols(2).leftCols(1);
  }
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelResidualTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::print(std::ostream& os) const {
  os << "CostModelResidual {" << *residual_ << ", " << *activation_ << "}";
}

}  // namespace crocoddyl
