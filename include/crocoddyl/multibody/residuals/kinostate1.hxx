///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/kinostate1.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl
{

  template <typename Scalar>
  ResidualFlyState1Tpl<Scalar>::ResidualFlyState1Tpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref, const std::size_t nu)
      : Base(state, 3, nu, false, false, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyState1Tpl<Scalar>::ResidualFlyState1Tpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref)
      : Base(state, 3, false, false, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyState1Tpl<Scalar>::ResidualFlyState1Tpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const std::size_t nu)
      : Base(state, 3, nu, false, false, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyState1Tpl<Scalar>::ResidualFlyState1Tpl(boost::shared_ptr<typename Base::StateAbstract> state)
      : Base(state, 3, false, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyState1Tpl<Scalar>::~ResidualFlyState1Tpl() {}

  template <typename Scalar>
  void ResidualFlyState1Tpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                         const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->diff1(xref_, x, data->r); //diff1
    //data->r.setZero();
    data->r.head(2) = x.head(21).tail(2);
    data->r.tail(1) = x.tail(1);
   
    
    //td::cout << "x_ref " << std::endl;
    //std::cout << xref_ << std::endl; // size 52

    //data->r.tail(1) = x.head(state_->get_nq()).tail(1);
    //xref_ = data->r;
  }

  template <typename Scalar>
  void ResidualFlyState1Tpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                             const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->Jdiff1(xref_, x, data->Rx, data->Rx, second);//diff1

    //data->Rx.setZero();
    //data->Rx.bottomLeftCorner(2, state_->get_nq()-1).topRightCorner(1, 1).diagonal().array() = (Scalar)1;
    data->Rx.bottomLeftCorner(2, 20).bottomRightCorner(2, 2).diagonal().array() = (Scalar)1;
    data->Rx.bottomRightCorner(1, 1).diagonal().array() = (Scalar)1;

   
  }

  template <typename Scalar>
  void ResidualFlyState1Tpl<Scalar>::print(std::ostream &os) const
  {
    os << "ResidualFlyState";
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &ResidualFlyState1Tpl<Scalar>::get_reference() const
  {
    return xref_;
  }

  template <typename Scalar>
  void ResidualFlyState1Tpl<Scalar>::set_reference(const VectorXs &reference)
  {
    xref_ = reference;
  }
} // namespace crocoddyl

